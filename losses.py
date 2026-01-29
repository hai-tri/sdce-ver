"""
Custom Cross-Entropy Loss with Surrogate Guidance

This module provides loss functions for language model training with optional
surrogate model guidance.

TPU/MPS Compatibility Notes:
- TPU does not support float16 (fp16), only bfloat16 and float32
- Operations like softmax, log_softmax, and reciprocal can produce NaN in lower precision
- All numerically sensitive operations are computed in float32 for stability
- Results are cast back to the input dtype when safe to do so
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


# =============================================================================
# TPU-Safe Tensor Operations
# =============================================================================
# These helpers ensure numerical stability on TPU/MPS/float16 by computing
# sensitive operations in float32. This prevents NaN/Inf issues that commonly
# occur with lower precision formats.


def safe_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute softmax in float32 for numerical stability.

    TPU and MPS can produce NaN values when computing softmax in float16/bfloat16,
    especially when logits contain extreme values or -inf (from masking).

    Args:
        logits: Input tensor of any dtype
        dim: Dimension along which to compute softmax

    Returns:
        Softmax probabilities in the same dtype as input
    """
    input_dtype = logits.dtype
    return F.softmax(logits.float(), dim=dim).to(input_dtype)


def safe_log_softmax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """Compute log_softmax in float32 for numerical stability.

    Log-softmax is particularly sensitive to overflow/underflow because it involves
    both exp() and log() operations. Computing in float32 prevents NaN on TPU/MPS.

    Args:
        logits: Input tensor of any dtype
        dim: Dimension along which to compute log_softmax

    Returns:
        Log-softmax values. Returns float32 to preserve precision for subsequent
        operations (e.g., NLL computation).
    """
    # Keep output in float32 since log values are often used in further computations
    return F.log_softmax(logits.float(), dim=dim)


def safe_reciprocal(x: torch.Tensor, eps: float = 1e-6, max_val: float = 1e6) -> torch.Tensor:
    """Compute reciprocal (1/x) safely with clamping.

    Reciprocal operations are prone to overflow (when x is very small) and
    produce inf values. This helper adds epsilon before the reciprocal and
    clamps the result to prevent extreme values.

    Args:
        x: Input tensor (typically probabilities)
        eps: Small epsilon added to prevent division by zero
        max_val: Maximum value to clamp the result to

    Returns:
        Reciprocal values in float32, clamped to [0, max_val]
    """
    return torch.reciprocal(x.float() + eps).clamp(max=max_val)


def safe_weighted_sum(
    values: torch.Tensor,
    weights: torch.Tensor,
    handle_inf: bool = True
) -> torch.Tensor:
    """Compute weighted sum safely, handling inf * 0 = NaN case.

    When values contain inf (e.g., from -log(0)) and weights contain 0
    (e.g., from masking), the product inf * 0 = NaN. This helper zeros out
    values where weights are zero BEFORE multiplication.

    Args:
        values: Values to weight (may contain inf)
        weights: Weights to apply (may contain 0)
        handle_inf: If True, zero out values where weights are 0

    Returns:
        Weighted values (same shape as input), safe from inf * 0 = NaN
    """
    if handle_inf:
        # Zero out values where weights are zero to avoid inf * 0 = NaN
        safe_values = torch.where(
            weights > 0,
            values,
            torch.zeros_like(values)
        )
        return safe_values * weights
    return values * weights


def surrogate_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    perp_values: Optional[torch.Tensor] = None,
    perp_indices: Optional[torch.Tensor] = None,
    lookup_surrogate_to_self: Optional[torch.Tensor] = None,
    surrogate_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    use_perplexity_weighting: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
    """
    Cross-entropy loss with optional surrogate-guided auxiliary term.
    
    This function computes the standard cross-entropy loss and optionally adds
    a surrogate-guided term that encourages the model to assign probability mass
    to tokens that the surrogate model finds important.
    
    Args:
        logits: Model output logits. Shape can be:
            - (batch_size * seq_len, vocab_size) for flattened inputs, or
            - (batch_size, seq_len, vocab_size) when using surrogate guidance
        labels: Target labels. Shape matches logits batch dimensions.
        perp_values: Perplexity values from surrogate model.
            Shape: (batch_size, seq_len, k) where k is the number of top tokens
        perp_indices: Token indices from surrogate model (in surrogate vocab).
            Shape: (batch_size, seq_len, k)
        lookup_surrogate_to_self: Lookup table mapping surrogate vocabulary
            indices to base model vocabulary indices. Shape: (surrogate_vocab_size,)
        surrogate_weight: Weight for the surrogate loss term (decays over training via cosine schedule)
        ignore_index: Index to ignore in loss computation (typically -100 for padding)
        reduction: How to reduce the loss - "mean", "sum", or "none"
        compute_z_loss: Whether to compute auxiliary z-loss (from PaLM paper)
        z_loss_multiplier: Coefficient for z-loss term
        use_perplexity_weighting: If True (default), weight tokens by softmax(-perplexity).
            If False, use uniform weights of 1 for all valid masked tokens.
        
    Returns:
        Tuple of (loss, z_loss, num_aux_tokens) where:
        - z_loss is None if compute_z_loss=False
        - num_aux_tokens is the count of valid auxiliary tokens used (None if no surrogate)
        
    Example:
        >>> # Standard cross-entropy (no surrogate)
        >>> loss, _, _ = surrogate_cross_entropy_loss(logits, labels)
        
        >>> # With surrogate guidance
        >>> loss, z_loss, num_aux = surrogate_cross_entropy_loss(
        ...     logits, labels,
        ...     perp_values=surr_perp_values,
        ...     perp_indices=surr_perp_indices,
        ...     lookup_surrogate_to_self=lookup_table,
        ...     surrogate_weight=0.5,  # Decayed weight
        ...     compute_z_loss=True
        ... )
    """
    device = logits.device
    num_aux_tokens = None  # Will be set if using surrogate guidance
    
    if perp_indices is not None and lookup_surrogate_to_self is not None and surrogate_weight > 0:
        # =====================================================================
        # Surrogate-guided loss computation
        # =====================================================================
        
        # Get dimensions
        if logits.dim() == 2:
            # Need to infer batch/seq dimensions from perp_indices
            batch_size, seq_len, k = perp_indices.shape
            vocab_size = logits.shape[-1]
            logits = logits.view(batch_size, seq_len, vocab_size)
            labels = labels.view(batch_size, seq_len)
        else:
            batch_size, seq_len, vocab_size = logits.shape
            k = perp_indices.shape[-1]
        
        # Translate surrogate indices to base model vocabulary
        # Handle out-of-bounds by clamping (invalid translations remain -100)
        translated_perp_indices = lookup_surrogate_to_self[perp_indices]
        
        # Create mask for valid translations (not -100 and within vocab)
        valid_translation_mask = (translated_perp_indices >= 0) & (translated_perp_indices < vocab_size)
        
        # Clamp to valid range for gather operation (will be masked out anyway)
        safe_translated_indices = translated_perp_indices.clamp(0, vocab_size - 1)
        
        # Gather log probabilities for the translated top-k tokens
        # IMPORTANT: Compute log_softmax over FULL vocabulary first, then gather.
        # Computing softmax only over the k gathered logits would produce incorrect
        # probabilities because softmax normalizes over all classes. For example:
        #   Full vocab logits [10, 5, 3, 2, 1] -> softmax -> [0.99, 0.006, 0.001, ...]
        #   Gathering [5, 3] then softmax -> [0.88, 0.12] (WRONG!)
        # The correct approach: softmax full vocab, then gather the probabilities.
        # Use safe_log_softmax for TPU/MPS numerical stability
        full_log_probs = safe_log_softmax(logits, dim=-1)
        gathered_log_probs = torch.gather(full_log_probs, dim=2, index=safe_translated_indices)
        gathered_nll = -gathered_log_probs  # Negative log likelihood (in float32)
        
        # === BUILD VALIDITY MASK ===
        # A token is valid if:
        # 1. Its translation to base vocab succeeded (valid_translation_mask)
        # 2. Its perplexity is finite (not masked by probability threshold)
        # 3. The entire row isn't invalid (at least one token should be valid)
        
        # Check individual token validity (finite perplexity = above probability threshold)
        perp_finite_mask = ~torch.isinf(perp_values)  # (batch, seq, k)
        
        # Combine: valid translation AND finite perplexity
        token_valid_mask = valid_translation_mask & perp_finite_mask  # (batch, seq, k)
        
        # Check row validity (at least one token in the row is valid)
        row_has_valid_token = token_valid_mask.any(dim=-1)  # (batch, seq)
        
        # Final mask: token is valid AND row is valid
        valid_row_mask = token_valid_mask & row_has_valid_token.unsqueeze(-1)  # (batch, seq, k)
        
        # Count valid entries for normalization and logging
        num_valid_surrogate_entries = valid_row_mask.sum().item()
        num_aux_tokens = int(num_valid_surrogate_entries)  # For return value
        
        # Compute softmax weights from perplexity (lower perp = higher weight)
        # We compute softmax(-perplexity) so lower perplexity -> higher weight
        # 
        # CRITICAL: Handle rows where ALL tokens are invalid (all perp = inf).
        # softmax([-inf, -inf, -inf]) = [0/0, 0/0, 0/0] = [NaN, NaN, NaN]
        # 
        # Solution: For rows with no valid tokens, replace with zeros BEFORE softmax.
        # The softmax of zeros is uniform [1/k, 1/k, ...], but we multiply by
        # valid_row_mask afterwards which zeros out these invalid rows anyway.
        masked_perp_values = perp_values.clone()
        masked_perp_values[~valid_row_mask] = float('inf')
        
        # For rows with NO valid tokens, replace entire row with zeros to avoid NaN
        # (softmax of all -inf is NaN, but softmax of all zeros is valid)
        row_has_valid = row_has_valid_token.unsqueeze(-1).expand_as(masked_perp_values)
        masked_perp_values = torch.where(row_has_valid, masked_perp_values, torch.zeros_like(masked_perp_values))
        
        # Compute weights based on use_perplexity_weighting flag
        if use_perplexity_weighting:
            # Use safe_softmax for TPU/MPS numerical stability with -inf inputs
            softmax_weights = safe_softmax(-masked_perp_values, dim=-1)
            softmax_weights = softmax_weights * valid_row_mask.float()  # Zero out invalid positions
        else:
            # Use uniform weights: 1 for valid positions, 0 for invalid
            softmax_weights = valid_row_mask.float()

        # Weighted surrogate loss (scaled by surrogate_weight)
        # Use safe_weighted_sum to handle inf * 0 = NaN case
        weighted_nll = safe_weighted_sum(gathered_nll, softmax_weights, handle_inf=True)
        surrogate_loss = surrogate_weight * weighted_nll.sum()
        
        # Standard cross-entropy loss
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        ce_loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction='sum')
        
        # Combine losses
        num_valid_labels = (labels != ignore_index).sum().item()
        
        # Normalization: CE loss normalized by labels, surrogate loss normalized by surrogate entries
        # Then combined (surrogate already scaled by surrogate_weight)
        if reduction == "mean":
            ce_loss_normalized = ce_loss / (num_valid_labels + 1e-8)
            surrogate_loss_normalized = surrogate_loss / (num_valid_surrogate_entries + 1e-8)
            loss = ce_loss_normalized + surrogate_loss_normalized
        elif reduction == "sum":
            loss = ce_loss + surrogate_loss
        else:  # reduction == "none"
            # Return per-token loss (reshape surrogate loss appropriately)
            ce_loss_per_token = F.cross_entropy(
                logits_flat, labels_flat, ignore_index=ignore_index, reduction='none'
            ).view(batch_size, seq_len)
            surr_loss_per_token = surrogate_weight * weighted_nll.sum(dim=-1)
            loss = ce_loss_per_token + surr_loss_per_token
        
    else:
        # =====================================================================
        # Standard cross-entropy loss (no surrogate guidance)
        # =====================================================================
        if logits.dim() == 3:
            logits = logits.view(-1, logits.size(-1))
            labels = labels.view(-1)
        
        loss = F.cross_entropy(
            logits, labels,
            ignore_index=ignore_index,
            reduction=reduction
        )
    
    # =========================================================================
    # Optional Z-loss computation (from PaLM paper)
    # =========================================================================
    z_loss = None
    if compute_z_loss:
        # Z-loss penalizes large logits to stabilize training
        # z = logsumexp(logits) => z_loss = z^2
        if logits.dim() == 2:
            z_squared = logits.logsumexp(dim=-1).pow(2)
            label_mask = (labels != ignore_index).float()
        else:
            z_squared = logits.view(-1, logits.size(-1)).logsumexp(dim=-1).pow(2)
            label_mask = (labels.view(-1) != ignore_index).float()
        
        if reduction == "mean":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum() / (label_mask.sum() + 1e-8)
        elif reduction == "sum":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum()
        else:  # reduction == "none"
            z_loss = z_loss_multiplier * z_squared
    
    return loss, z_loss, num_aux_tokens


def standard_cross_entropy_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Standard cross-entropy loss for typical language model training.
    
    Args:
        logits: Model output logits (batch_size, seq_len, vocab_size) or (batch_size * seq_len, vocab_size)
        labels: Target labels (batch_size, seq_len) or (batch_size * seq_len,)
        ignore_index: Index to ignore in loss computation
        reduction: "mean", "sum", or "none"
        compute_z_loss: Whether to compute auxiliary z-loss
        z_loss_multiplier: Multiplier for z-loss
        
    Returns:
        Tuple of (loss, z_loss) where z_loss may be None
    """
    # Flatten if needed
    if logits.dim() == 3:
        vocab_size = logits.shape[-1]
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
    else:
        vocab_size = logits.shape[-1]
        logits_flat = logits
        labels_flat = labels
    
    # Standard cross-entropy
    loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=ignore_index, reduction=reduction)
    
    # Compute z-loss if requested
    z_loss = None
    if compute_z_loss:
        z_squared = logits_flat.logsumexp(-1).pow(2)
        label_mask = (labels_flat != ignore_index).float()
        
        if reduction == "mean":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum() / (label_mask.sum() + 1e-8)
        elif reduction == "sum":
            z_loss = z_loss_multiplier * (z_squared * label_mask).sum()
        else:
            z_loss = z_loss_multiplier * z_squared
    
    return loss, z_loss


def kl_divergence_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 1.0,
    kl_weight: float = 1.0,
    ignore_index: int = -100,
    reduction: str = "mean",
    compute_z_loss: bool = False,
    z_loss_multiplier: float = 1e-4,
    student_vocab_indices: Optional[torch.Tensor] = None,
    teacher_vocab_indices: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Knowledge distillation loss combining cross-entropy with KL divergence.

    This loss encourages the student model to match the teacher (surrogate) model's
    output distribution while still learning from the ground truth labels.

    Loss = CE(student, labels) + kl_weight * KL(teacher || student)

    When student and teacher have different vocabularies, the KL divergence is
    computed only over the intersection of vocabularies using the provided
    index mappings.

    Args:
        student_logits: Student model logits. Shape: (batch_size, seq_len, student_vocab_size)
            or (batch_size * seq_len, student_vocab_size)
        teacher_logits: Teacher model logits. Shape: (batch_size, seq_len, teacher_vocab_size)
            or (batch_size * seq_len, teacher_vocab_size)
        labels: Target labels (in student vocab). Shape: (batch_size, seq_len) or (batch_size * seq_len,)
        temperature: Temperature for softening distributions (higher = softer)
        kl_weight: Weight for the KL divergence term
        ignore_index: Index to ignore in loss computation (typically -100 for padding)
        reduction: How to reduce the loss - "mean", "sum", or "none"
        compute_z_loss: Whether to compute auxiliary z-loss
        z_loss_multiplier: Multiplier for z-loss
        student_vocab_indices: Indices in student vocab that exist in intersection.
            Shape: (intersection_size,). If None, assumes identical vocabularies.
        teacher_vocab_indices: Indices in teacher vocab that exist in intersection.
            Shape: (intersection_size,). If None, assumes identical vocabularies.

    Returns:
        Tuple of (combined_loss, z_loss) where z_loss may be None

    Note:
        The KL divergence is computed as KL(P_teacher || P_student) which encourages
        the student to cover modes of the teacher distribution.
    """
    # Flatten if needed
    if student_logits.dim() == 3:
        batch_size, seq_len, student_vocab_size = student_logits.shape
        student_logits_flat = student_logits.view(-1, student_vocab_size)
        labels_flat = labels.view(-1)
    else:
        student_vocab_size = student_logits.shape[-1]
        student_logits_flat = student_logits
        labels_flat = labels

    if teacher_logits.dim() == 3:
        teacher_logits_flat = teacher_logits.view(-1, teacher_logits.shape[-1])
    else:
        teacher_logits_flat = teacher_logits

    # Create mask for valid tokens (not padding)
    valid_mask = (labels_flat != ignore_index)

    # Standard cross-entropy loss on valid tokens (full student vocab)
    ce_loss = F.cross_entropy(
        student_logits_flat,
        labels_flat,
        ignore_index=ignore_index,
        reduction=reduction,
    )

    # KL divergence: KL(P_teacher || P_student)
    # If vocabularies differ, compute KL only over the intersection
    if student_vocab_indices is not None and teacher_vocab_indices is not None:
        # Extract logits only for intersection tokens
        # student_vocab_indices: indices in student vocab for shared tokens
        # teacher_vocab_indices: indices in teacher vocab for shared tokens
        student_intersection_logits = student_logits_flat[:, student_vocab_indices]
        teacher_intersection_logits = teacher_logits_flat[:, teacher_vocab_indices]

        # Apply temperature scaling and compute probabilities over intersection
        student_log_probs = F.log_softmax(student_intersection_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_intersection_logits / temperature, dim=-1)
    else:
        # Same vocabulary - use full logits
        student_log_probs = F.log_softmax(student_logits_flat / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_flat / temperature, dim=-1)

    # KL divergence per token: sum over vocab dimension
    # KL(P || Q) = sum(P * log(P/Q)) = sum(P * log(P)) - sum(P * log(Q))
    kl_per_token = F.kl_div(
        student_log_probs,
        teacher_probs,
        reduction="none",
    ).sum(dim=-1)  # Shape: (batch_size * seq_len,)

    # Apply mask and reduce
    if reduction == "mean":
        num_valid = valid_mask.sum().float().clamp(min=1.0)
        kl_loss = (kl_per_token * valid_mask.float()).sum() / num_valid
    elif reduction == "sum":
        kl_loss = (kl_per_token * valid_mask.float()).sum()
    else:
        kl_loss = kl_per_token * valid_mask.float()

    # Scale KL loss by temperature^2 (standard practice in knowledge distillation)
    # This accounts for the gradient magnitude change from temperature scaling
    kl_loss = kl_loss * (temperature ** 2)

    # Combined loss
    loss = ce_loss + kl_weight * kl_loss

    # Compute z-loss if requested
    z_loss = None
    if compute_z_loss:
        z_squared = student_logits_flat.logsumexp(-1).pow(2)

        if reduction == "mean":
            z_loss = z_loss_multiplier * (z_squared * valid_mask.float()).sum() / valid_mask.sum().float().clamp(min=1.0)
        elif reduction == "sum":
            z_loss = z_loss_multiplier * (z_squared * valid_mask.float()).sum()
        else:
            z_loss = z_loss_multiplier * z_squared

    return loss, z_loss


def compute_intersection_attention_mask(
    input_ids: torch.Tensor,
    lookup_base_to_surrogate: torch.Tensor,
) -> torch.Tensor:
    """
    Compute attention mask that masks out tokens not in the vocabulary intersection.
    
    Tokens that don't exist in the surrogate vocabulary will be masked out (0),
    so the surrogate model's attention mechanism ignores them.
    
    Args:
        input_ids: Input token IDs in base model vocabulary. Shape: (batch, seq_len)
        lookup_base_to_surrogate: Lookup table from base vocab to surrogate vocab.
            Contains -100 for tokens not in surrogate vocab.
            
    Returns:
        attention_mask: Binary mask where 1 = token in intersection, 0 = not in intersection.
            Shape: (batch, seq_len)
    """
    # Translate input tokens to surrogate vocab
    # Tokens not in surrogate vocab will have value -100
    translated_ids = lookup_base_to_surrogate[input_ids]
    
    # Create mask: 1 if token exists in intersection, 0 otherwise
    attention_mask = (translated_ids != -100).to(input_ids.dtype)
    
    return attention_mask


def compute_perplexity_guidance(
    surrogate_logits: torch.Tensor,
    labels: torch.Tensor,
    lookup_base_to_surrogate: torch.Tensor,
    permitted_surrogate_ids: torch.Tensor,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute top-k perplexity-based guidance from surrogate model.
    
    Args:
        surrogate_logits: Logits from surrogate model. Shape: (batch, seq_len, surrogate_vocab)
        labels: Target labels in base model vocabulary. Shape: (batch, seq_len)
        lookup_base_to_surrogate: Lookup table from base vocab to surrogate vocab
        permitted_surrogate_ids: Surrogate token IDs that are in vocabulary intersection
        k: Number of top tokens to select
        
    Returns:
        Tuple of:
        - perp_values: Perplexity values for top-k tokens. Shape: (batch, seq_len, k)
            Entire row is inf if target token doesn't exist in intersection.
        - perp_indices: Surrogate vocab indices for top-k tokens. Shape: (batch, seq_len, k)
    """
    device = surrogate_logits.device
    batch_size, seq_len, vocab_size = surrogate_logits.shape

    # Compute perplexity (reciprocal of probability)
    # Use safe helpers for TPU/MPS numerical stability
    surrogate_probs = safe_softmax(surrogate_logits, dim=-1)
    surrogate_perp = safe_reciprocal(surrogate_probs, eps=1e-6, max_val=1e6)
    
    # Mask tokens not in vocabulary intersection
    vocab_indices = torch.arange(vocab_size, device=device)
    not_in_intersection = ~torch.isin(vocab_indices, permitted_surrogate_ids)
    surrogate_perp[:, :, not_in_intersection] = float('inf')
    
    # Mask positions where labels are invalid (-100)
    invalid_label_mask = (labels == -100).unsqueeze(-1)
    surrogate_perp = surrogate_perp.masked_fill(invalid_label_mask, float('inf'))
    
    # Translate labels to surrogate vocab
    safe_labels = labels.clone()
    safe_labels[labels == -100] = 0
    translated_labels = lookup_base_to_surrogate[safe_labels]
    
    # Check if target tokens exist in the intersection
    # If a target doesn't exist (translated to -100), zero out entire row's contribution
    target_not_in_intersection = (translated_labels == -100) & (labels != -100)
    target_not_in_intersection_mask = target_not_in_intersection.unsqueeze(-1)
    surrogate_perp = surrogate_perp.masked_fill(target_not_in_intersection_mask, float('inf'))
    
    # Create indices for masking out the actual label tokens
    valid_translation_mask = translated_labels != -100
    batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, seq_len)
    seq_idx = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    
    # Create safe indices for scatter (use 0 for invalid translations)
    safe_translated = translated_labels.clone()
    safe_translated[~valid_translation_mask] = 0
    
    # Mask out the actual labels (we don't want to include the target in top-k)
    # Only do this where the translation is valid
    surrogate_perp[batch_idx[valid_translation_mask], 
                   seq_idx[valid_translation_mask], 
                   safe_translated[valid_translation_mask]] = float('inf')
    
    # Get top-k lowest perplexity (highest confidence) tokens
    topk_result = torch.topk(surrogate_perp, k=k, largest=False, sorted=True, dim=-1)
    
    return topk_result.values, topk_result.indices


class SurrogateCrossEntropyLoss(torch.nn.Module):
    """
    PyTorch module wrapper for surrogate-guided cross-entropy loss.
    
    Example:
        >>> criterion = SurrogateCrossEntropyLoss(
        ...     lookup_table=lookup_surrogate_to_base,
        ...     compute_z_loss=True
        ... )
        >>> loss, z_loss, num_aux = criterion(logits, labels, perp_values, perp_indices)
    """
    
    def __init__(
        self,
        lookup_table: Optional[torch.Tensor] = None,
        ignore_index: int = -100,
        reduction: str = "mean",
        compute_z_loss: bool = False,
        z_loss_multiplier: float = 1e-4,
        use_perplexity_weighting: bool = True,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.compute_z_loss = compute_z_loss
        self.z_loss_multiplier = z_loss_multiplier
        self.use_perplexity_weighting = use_perplexity_weighting
        
        if lookup_table is not None:
            self.register_buffer('lookup_table', lookup_table)
        else:
            self.lookup_table = None
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        perp_values: Optional[torch.Tensor] = None,
        perp_indices: Optional[torch.Tensor] = None,
        surrogate_weight: float = 1.0,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[int]]:
        return surrogate_cross_entropy_loss(
            logits=logits,
            labels=labels,
            perp_values=perp_values,
            perp_indices=perp_indices,
            lookup_surrogate_to_self=self.lookup_table,
            surrogate_weight=surrogate_weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            compute_z_loss=self.compute_z_loss,
            z_loss_multiplier=self.z_loss_multiplier,
            use_perplexity_weighting=self.use_perplexity_weighting,
        )
