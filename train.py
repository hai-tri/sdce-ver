#!/usr/bin/env python3
"""
Surrogate-Assisted Language Model Training Script

This script implements a training approach where a surrogate model guides the 
primary model's learning by providing token-level perplexity signals.

Supports: CUDA GPUs, Apple MPS, TPUs (via PyTorch XLA), and CPU.

Features:
    - Train from scratch: Initialize model with random weights instead of pretrained
    - Auto eval split: Automatically creates evaluation split if not available
    - torchrun DDP: Distributed data parallel training via torchrun
    - TPU support: Native PyTorch XLA support for Google Cloud TPUs
    - TPU Pod support: Multi-host TPU training for TPU v3-32, v4-64, etc.

Usage:
    # Single GPU/CPU training
    python train.py --config config.yaml
    python train.py --base_model gpt2 --surrogate_model Qwen/Qwen3-0.6B --dataset wikitext --dataset_config wikitext-2-raw-v1
    
    # Train from scratch (random initialization)
    python train.py --config config.yaml --init_from_scratch
    
    # Multi-GPU training with torchrun
    torchrun --nproc_per_node=4 train.py --config config.yaml
    torchrun --nproc_per_node=auto train.py --config config.yaml  # all available GPUs
    
    # TPU training (single core)
    python train.py --config config.yaml --device tpu
    
    # TPU training (single host, multi-core with xmp.spawn)
    python train.py --config config.yaml --device tpu --tpu_cores 8
    
    # TPU Pod training (multi-host, e.g., v3-32 with 4 hosts × 8 cores)
    # Run on each host with proper TPU_WORKER_ID environment variable
    python train.py --config config.yaml --device tpu --tpu_cores 8 --tpu_num_hosts 4
    
    # TPU Pod training with PJRT runtime (recommended for PyTorch XLA 2.0+)
    # Set environment variables on each host:
    #   export PJRT_DEVICE=TPU
    #   export TPU_PROCESS_ADDRESSES=host1:port,host2:port,...
    #   export TPU_PROCESS_COUNT=<num_hosts>
    #   export TPU_PROCESS_ID=<this_host_id>
    python train.py --config config.yaml --device tpu --tpu_cores 8 --tpu_num_hosts 4
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm

# Optional: CUDA AMP
try:
    from torch.cuda.amp import GradScaler, autocast
    HAS_CUDA_AMP = True
except ImportError:
    HAS_CUDA_AMP = False
    GradScaler = None
    autocast = None

# Optional: PyTorch XLA for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.parallel_loader as pl
    import torch_xla.distributed.xla_multiprocessing as xmp
    HAS_XLA = True
    
    # Try to import PJRT runtime for TPU pod support (PyTorch XLA 2.0+)
    try:
        import torch_xla.runtime as xr
        HAS_XLA_RUNTIME = True
    except ImportError:
        HAS_XLA_RUNTIME = False
        xr = None
    
    # Try to import rendezvous for multi-host coordination
    try:
        import torch_xla.distributed.xla_backend
        HAS_XLA_BACKEND = True
    except ImportError:
        HAS_XLA_BACKEND = False
        
except ImportError:
    HAS_XLA = False
    HAS_XLA_RUNTIME = False
    HAS_XLA_BACKEND = False
    xm = None
    pl = None
    xmp = None
    xr = None

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_scheduler,
)

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False

# lm-evaluation-harness for benchmarking
try:
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.evaluator import simple_evaluate
    HAS_LM_EVAL = True
except ImportError:
    HAS_LM_EVAL = False
    lm_eval = None
    HFLM = None
    simple_evaluate = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Import loss functions from losses module
from losses import (
    surrogate_cross_entropy_loss,
    standard_cross_entropy_loss,
    kl_divergence_loss,
    compute_intersection_attention_mask,
    compute_perplexity_guidance,
    SurrogateCrossEntropyLoss,
)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model initialization."""
    name_or_path: str = "gpt2"
    dtype: str = "float16"  # float16, bfloat16, float32
    use_flash_attention: bool = False
    gradient_checkpointing: bool = False
    trust_remote_code: bool = False
    init_from_scratch: bool = False  # If True, initialize model with random weights
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)


@dataclass
class SurrogateConfig:
    """Configuration for the surrogate model."""
    name_or_path: str = "Qwen/Qwen3-0.6B"
    dtype: str = "float16"
    k: int = 30  # Number of top-k tokens to select by probability (candidate pool)
    probability_threshold: float = 0.02  # Min probability to include token (tokens below are masked out)
    enabled: bool = True
    trust_remote_code: bool = False
    loss_weight_initial: float = 1.0  # Initial weight for surrogate loss
    loss_weight_final: float = 0.0    # Final weight for surrogate loss (after cosine decay)
    use_perplexity_weighting: bool = True  # If True, weight by softmax(-perplexity); if False, use uniform weights
    
    def get_torch_dtype(self) -> torch.dtype:
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)


@dataclass
class DataConfig:
    """Configuration for data loading."""
    dataset_name: str = "wikitext"
    dataset_config: Optional[str] = "wikitext-2-raw-v1"
    dataset_split: str = "train"
    eval_split: str = "validation"
    text_column: str = "text"
    max_seq_length: int = 1024
    preprocessing_num_workers: int = 4
    train_file: Optional[str] = None  # For custom datasets
    eval_file: Optional[str] = None
    eval_split_ratio: float = 0.05  # Ratio of train data to use for eval if no eval split exists
    eval_split_seed: int = 42  # Seed for reproducible eval split creation


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = "./outputs"
    num_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    warmup_steps: Optional[int] = None
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"
    
    # Loss type: "standard" (CE only), "surrogate" (CE + surrogate-guided), "kl" (CE + KL divergence)
    loss_type: str = "surrogate"
    
    # Device selection: "auto", "cuda", "mps", "tpu", "cpu"
    device: str = "auto"
    
    # TPU-specific options
    tpu_cores: int = 1  # Number of TPU cores per host (1 for single, 8 for v3-8, etc.)
    tpu_num_hosts: int = 1  # Number of TPU hosts (1 for single host, >1 for TPU pods)
    tpu_metrics_debug: bool = False  # Print TPU metrics for debugging
    tpu_use_pjrt: bool = True  # Use PJRT runtime (recommended for TPU pods)
    
    # Precision
    mixed_precision: str = "fp16"  # fp16, bf16, no
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 500
    save_steps: int = 1000
    save_total_limit: int = 3
    
    # Auxiliary loss
    use_z_loss: bool = False
    z_loss_multiplier: float = 1e-4
    
    # Distributed (for DDP on GPU)
    local_rank: int = -1
    
    # Misc
    seed: int = 42
    resume_from_checkpoint: Optional[str] = None
    
    # W&B
    wandb_project: Optional[str] = None
    wandb_run_name: Optional[str] = None
    wandb_entity: Optional[str] = None


@dataclass
class EvaluationConfig:
    """Configuration for lm-evaluation-harness benchmarks."""
    enabled: bool = True
    eval_interval: int = 1000  # Run benchmarks every N steps
    
    # Benchmark tasks to run (from lm-evaluation-harness)
    # Common options: "hellaswag", "arc_easy", "arc_challenge", "winogrande", 
    # "piqa", "boolq", "lambada_openai", "mmlu", "truthfulqa_mc"
    tasks: List[str] = field(default_factory=lambda: ["hellaswag", "arc_easy", "piqa"])
    
    # Number of few-shot examples (0 for zero-shot)
    num_fewshot: int = 0
    
    # Batch size for evaluation
    batch_size: int = 8
    
    # Limit number of examples per task (None for all)
    limit: Optional[int] = None
    
    # Log individual task scores
    log_individual_tasks: bool = True
    
    # Log aggregated score (mean across tasks)
    log_aggregate_score: bool = True


@dataclass
class Config:
    """Main configuration combining all sub-configs."""
    model: ModelConfig = field(default_factory=ModelConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    def _sync_surrogate_enabled(self) -> None:
        """
        Ensure surrogate.enabled is consistent with training.loss_type.
        
        The loss_type is the authoritative source:
        - "standard": surrogate.enabled = False (no surrogate needed)
        - "surrogate": surrogate.enabled = True (SDCE loss needs surrogate)
        - "kl": surrogate.enabled = True (KL divergence needs surrogate)
        
        This prevents configuration errors where loss_type and surrogate.enabled
        are inconsistent.
        """
        if self.training.loss_type == "standard":
            self.surrogate.enabled = False
        elif self.training.loss_type in ("surrogate", "kl"):
            self.surrogate.enabled = True
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        """Load configuration from YAML file."""
        if not HAS_YAML:
            raise ImportError("PyYAML is required to load config from YAML. Install with: pip install pyyaml")
        
        with open(path, "r") as f:
            raw_config = yaml.safe_load(f)
        
        config = cls(
            model=ModelConfig(**raw_config.get("model", {})),
            surrogate=SurrogateConfig(**raw_config.get("surrogate", {})),
            data=DataConfig(**raw_config.get("data", {})),
            training=TrainingConfig(**raw_config.get("training", {})),
            evaluation=EvaluationConfig(**raw_config.get("evaluation", {})),
        )
        # Ensure surrogate.enabled is consistent with loss_type
        config._sync_surrogate_enabled()
        return config
    
    @classmethod
    def from_args(cls, args: argparse.Namespace) -> "Config":
        """Create configuration from command line arguments."""
        config = cls()
        
        # Model config
        if args.base_model:
            config.model.name_or_path = args.base_model
        if args.model_dtype:
            config.model.dtype = args.model_dtype
        if args.gradient_checkpointing:
            config.model.gradient_checkpointing = True
        if args.trust_remote_code:
            config.model.trust_remote_code = True
        if hasattr(args, 'init_from_scratch') and args.init_from_scratch:
            config.model.init_from_scratch = True
            
        # Surrogate config
        if args.surrogate_model:
            config.surrogate.name_or_path = args.surrogate_model
        if args.surrogate_k:
            config.surrogate.k = args.surrogate_k
        if args.no_surrogate:
            config.surrogate.enabled = False
        if args.surrogate_dtype:
            config.surrogate.dtype = args.surrogate_dtype
        if hasattr(args, 'probability_threshold') and args.probability_threshold is not None:
            config.surrogate.probability_threshold = args.probability_threshold
            
        # Data config
        if args.dataset:
            config.data.dataset_name = args.dataset
        if args.dataset_config:
            config.data.dataset_config = args.dataset_config
        if args.text_column:
            config.data.text_column = args.text_column
        if args.max_seq_length:
            config.data.max_seq_length = args.max_seq_length
        if args.train_file:
            config.data.train_file = args.train_file
        if args.eval_file:
            config.data.eval_file = args.eval_file
        if hasattr(args, 'eval_split_ratio') and args.eval_split_ratio:
            config.data.eval_split_ratio = args.eval_split_ratio
            
        # Training config
        if args.output_dir:
            config.training.output_dir = args.output_dir
        if args.num_epochs:
            config.training.num_epochs = args.num_epochs
        if args.batch_size:
            config.training.per_device_train_batch_size = args.batch_size
        if args.gradient_accumulation_steps:
            config.training.gradient_accumulation_steps = args.gradient_accumulation_steps
        if args.learning_rate:
            config.training.learning_rate = args.learning_rate
        if args.weight_decay:
            config.training.weight_decay = args.weight_decay
        if args.warmup_ratio:
            config.training.warmup_ratio = args.warmup_ratio
        if args.max_grad_norm:
            config.training.max_grad_norm = args.max_grad_norm
        if args.seed:
            config.training.seed = args.seed
        if args.mixed_precision:
            config.training.mixed_precision = args.mixed_precision
        if args.logging_steps:
            config.training.logging_steps = args.logging_steps
        if args.eval_steps:
            config.training.eval_steps = args.eval_steps
        if args.save_steps:
            config.training.save_steps = args.save_steps
        if args.wandb_project:
            config.training.wandb_project = args.wandb_project
        if args.wandb_run_name:
            config.training.wandb_run_name = args.wandb_run_name
        if args.resume_from_checkpoint:
            config.training.resume_from_checkpoint = args.resume_from_checkpoint
        if args.use_z_loss:
            config.training.use_z_loss = True
        if args.local_rank is not None:
            config.training.local_rank = args.local_rank
        if hasattr(args, 'device') and args.device:
            config.training.device = args.device
        if hasattr(args, 'tpu_cores') and args.tpu_cores:
            config.training.tpu_cores = args.tpu_cores
        
        # Loss type handling (convenience flags take precedence)
        if hasattr(args, 'standard_training') and args.standard_training:
            config.training.loss_type = "standard"
        elif hasattr(args, 'kl_divergence') and args.kl_divergence:
            config.training.loss_type = "kl"
        elif hasattr(args, 'loss_type') and args.loss_type:
            config.training.loss_type = args.loss_type
        
        # Ensure surrogate.enabled is consistent with loss_type
        config._sync_surrogate_enabled()
        return config
    
    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        config_dict = {
            "model": vars(self.model),
            "surrogate": vars(self.surrogate),
            "data": vars(self.data),
            "training": vars(self.training),
            "evaluation": vars(self.evaluation),
        }
        with open(path, "w") as f:
            json.dump(config_dict, f, indent=2)

# =============================================================================
# Metrics and FLOPS Computation
# =============================================================================

@dataclass
class SpeedMetrics:
    """Track training speed and throughput metrics."""
    start_time: float = 0.0
    total_start_time: float = 0.0  # Never reset - tracks total training time
    tokens_seen: int = 0
    batches_seen: int = 0
    total_tokens: int = 0  # Cumulative across all training

    def reset(self) -> None:
        """Reset metrics for a new logging interval."""
        self.start_time = time.time()
        self.tokens_seen = 0
        self.batches_seen = 0

    def start(self) -> None:
        """Initialize the total start time at the beginning of training."""
        self.total_start_time = time.time()
        self.start_time = time.time()

    def update(self, batch_tokens: int) -> None:
        """Update metrics with a new batch."""
        self.tokens_seen += batch_tokens
        self.batches_seen += 1
        self.total_tokens += batch_tokens

    def get_metrics(self, world_size: int = 1) -> Dict[str, float]:
        """Compute throughput metrics."""
        elapsed = time.time() - self.start_time
        if elapsed == 0:
            elapsed = 1e-8

        total_elapsed = time.time() - self.total_start_time
        if total_elapsed == 0:
            total_elapsed = 1e-8

        total_tokens_trained = self.total_tokens * world_size

        return {
            "tokens_per_second": self.tokens_seen / elapsed,
            "tokens_per_second_global": (self.tokens_seen * world_size) / elapsed,
            "batches_per_second": self.batches_seen / elapsed,
            "total_tokens_trained": total_tokens_trained,
            "total_tokens_per_second": total_tokens_trained / total_elapsed,
        }


def estimate_model_flops(
    model: PreTrainedModel,
    batch_size: int,
    seq_length: int,
) -> Tuple[int, int]:
    """
    Estimate FLOPs for a transformer model forward and backward pass.
    
    Based on: https://arxiv.org/abs/2001.08361 (Scaling Laws for Neural Language Models)
    Approximate FLOPs per token ≈ 6 * num_params (forward + backward)
    Forward only ≈ 2 * num_params per token
    
    Returns:
        Tuple of (flops_per_token, total_flops_per_batch)
    """
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    # Approximate FLOPs: 2 * params for forward, 4 * params for backward (2x forward)
    # Total = 6 * params per token
    flops_per_token_forward = 2 * num_params
    flops_per_token_backward = 4 * num_params
    flops_per_token_total = flops_per_token_forward + flops_per_token_backward
    
    # Total FLOPs per batch
    total_tokens = batch_size * seq_length
    total_flops = flops_per_token_total * total_tokens
    
    return flops_per_token_total, total_flops


# =============================================================================
# Token Vocabulary Alignment
# =============================================================================

class VocabularyAligner:
    """Handles vocabulary alignment between base and surrogate models."""
    
    def __init__(
        self,
        base_tokenizer: PreTrainedTokenizer,
        surrogate_tokenizer: PreTrainedTokenizer,
        device: torch.device,
    ):
        self.base_tokenizer = base_tokenizer
        self.surrogate_tokenizer = surrogate_tokenizer
        self.device = device
        
        # Build lookup tables
        self._build_lookup_tables()
        
    def _build_lookup_tables(self) -> None:
        """Build bidirectional lookup tables between vocabularies."""
        base_vocab = self.base_tokenizer.get_vocab()
        surrogate_vocab = self.surrogate_tokenizer.get_vocab()
        
        base_tokens = set(base_vocab.keys())
        surrogate_tokens = set(surrogate_vocab.keys())
        
        # Find intersection
        intersection = base_tokens.intersection(surrogate_tokens)
        logger.info(f"Vocabulary intersection size: {len(intersection)} tokens")
        logger.info(f"Base vocab size: {len(base_tokens)}, Surrogate vocab size: {len(surrogate_tokens)}")
        
        # Create lookup tables
        base_vocab_size = len(base_vocab)
        surrogate_vocab_size = len(surrogate_vocab)
        
        # Build index arrays on CPU first (much faster than individual GPU assignments)
        base_ids = []
        surrogate_ids = []
        for token in intersection:
            base_ids.append(base_vocab[token])
            surrogate_ids.append(surrogate_vocab[token])
        
        base_ids_tensor = torch.tensor(base_ids, dtype=torch.long)
        surrogate_ids_tensor = torch.tensor(surrogate_ids, dtype=torch.long)
        
        # Create lookup tables on CPU, then move to device
        # Base -> Surrogate
        self.lookup_base_to_surrogate = torch.full(
            (base_vocab_size,), fill_value=-100, dtype=torch.long
        )
        self.lookup_base_to_surrogate[base_ids_tensor] = surrogate_ids_tensor
        self.lookup_base_to_surrogate = self.lookup_base_to_surrogate.to(self.device)
        
        # Surrogate -> Base
        self.lookup_surrogate_to_base = torch.full(
            (surrogate_vocab_size,), fill_value=-100, dtype=torch.long
        )
        self.lookup_surrogate_to_base[surrogate_ids_tensor] = base_ids_tensor
        self.lookup_surrogate_to_base = self.lookup_surrogate_to_base.to(self.device)
        
        # Store permitted tokens
        self.base_permitted_ids = base_ids_tensor.to(self.device)
        self.surrogate_permitted_ids = surrogate_ids_tensor.to(self.device)
        
        logger.info("Lookup tables built successfully.")
        
    def translate_base_to_surrogate(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Translate base model token IDs to surrogate model token IDs."""
        return self.lookup_base_to_surrogate[token_ids.clamp(0, len(self.lookup_base_to_surrogate) - 1)]
    
    def translate_surrogate_to_base(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Translate surrogate model token IDs to base model token IDs."""
        return self.lookup_surrogate_to_base[token_ids.clamp(0, len(self.lookup_surrogate_to_base) - 1)]


# =============================================================================
# Dataset
# =============================================================================

class TextDataset(Dataset):
    """Dataset for language model training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Tokenize all texts
        logger.info("Tokenizing dataset...")
        self.examples = []
        for text in tqdm(texts, desc="Tokenizing"):
            if not text.strip():
                continue
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_tensors=None,
            )
            if len(encoded["input_ids"]) > 1:
                self.examples.append(encoded["input_ids"])
        
        logger.info(f"Created dataset with {len(self.examples)} examples")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"input_ids": torch.tensor(self.examples[idx], dtype=torch.long)}


def collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    pad_token_id: int,
    max_length: int,
) -> Dict[str, torch.Tensor]:
    """Collate function with padding."""
    input_ids = [item["input_ids"] for item in batch]
    
    # Pad to max length in batch
    max_len = min(max(len(ids) for ids in input_ids), max_length)
    
    padded_input_ids = []
    attention_masks = []
    
    for ids in input_ids:
        if len(ids) < max_len:
            padding_length = max_len - len(ids)
            padded_ids = torch.cat([ids, torch.full((padding_length,), pad_token_id, dtype=torch.long)])
            mask = torch.cat([torch.ones(len(ids), dtype=torch.long), torch.zeros(padding_length, dtype=torch.long)])
        else:
            padded_ids = ids[:max_len]
            mask = torch.ones(max_len, dtype=torch.long)
        
        padded_input_ids.append(padded_ids)
        attention_masks.append(mask)
    
    return {
        "input_ids": torch.stack(padded_input_ids),
        "attention_mask": torch.stack(attention_masks),
    }


def load_training_data(
    config: DataConfig,
    tokenizer: PreTrainedTokenizer,
) -> Tuple[Dataset, Optional[Dataset]]:
    """Load training and validation datasets.
    
    If no eval split exists in the HuggingFace dataset, automatically creates one
    by splitting the training data according to eval_split_ratio.
    """
    
    if config.train_file:
        # Load from local files
        logger.info(f"Loading data from local files: {config.train_file}")
        
        with open(config.train_file, "r") as f:
            if config.train_file.endswith(".json"):
                data = json.load(f)
                train_texts = [item.get(config.text_column, item) for item in data]
            else:
                train_texts = f.readlines()
        
        eval_texts = None
        if config.eval_file:
            with open(config.eval_file, "r") as f:
                if config.eval_file.endswith(".json"):
                    data = json.load(f)
                    eval_texts = [item.get(config.text_column, item) for item in data]
                else:
                    eval_texts = f.readlines()
        else:
            # Create eval split from train data if no eval file provided
            logger.info(f"No eval file provided, creating eval split from train data (ratio={config.eval_split_ratio})")
            train_texts, eval_texts = _create_eval_split(
                train_texts, 
                config.eval_split_ratio, 
                config.eval_split_seed
            )
    else:
        # Load from HuggingFace datasets
        if not HAS_DATASETS:
            raise ImportError("datasets library required. Install with: pip install datasets")
        
        logger.info(f"Loading dataset: {config.dataset_name} ({config.dataset_config})")
        
        dataset = load_dataset(
            config.dataset_name,
            config.dataset_config,
        )
        
        train_texts = dataset[config.dataset_split][config.text_column]
        eval_texts = None
        
        # Check for eval split with common names
        eval_split_names = [config.eval_split, "validation", "valid", "val", "test", "dev"]
        found_eval_split = None
        for split_name in eval_split_names:
            if split_name in dataset:
                found_eval_split = split_name
                break
        
        if found_eval_split is not None:
            logger.info(f"Using existing eval split: {found_eval_split}")
            eval_texts = dataset[found_eval_split][config.text_column]
        else:
            # No eval split found, create one from training data
            logger.info(
                f"No eval split found in dataset (tried: {eval_split_names}). "
                f"Creating eval split from train data (ratio={config.eval_split_ratio})"
            )
            train_texts, eval_texts = _create_eval_split(
                list(train_texts), 
                config.eval_split_ratio, 
                config.eval_split_seed
            )
    
    train_dataset = TextDataset(train_texts, tokenizer, config.max_seq_length)
    eval_dataset = TextDataset(eval_texts, tokenizer, config.max_seq_length) if eval_texts else None
    
    return train_dataset, eval_dataset


def _create_eval_split(
    texts: List[str],
    eval_ratio: float,
    seed: int,
) -> Tuple[List[str], List[str]]:
    """Split texts into train and eval sets.
    
    Args:
        texts: List of text samples
        eval_ratio: Fraction of data to use for evaluation (0.0 to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_texts, eval_texts)
    """
    if eval_ratio <= 0.0 or eval_ratio >= 1.0:
        logger.warning(f"Invalid eval_split_ratio={eval_ratio}, using 0.05")
        eval_ratio = 0.05
    
    # Set random seed for reproducibility
    rng = random.Random(seed)
    
    # Shuffle indices
    indices = list(range(len(texts)))
    rng.shuffle(indices)
    
    # Split
    split_idx = int(len(texts) * (1 - eval_ratio))
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]
    
    train_texts = [texts[i] for i in train_indices]
    eval_texts = [texts[i] for i in eval_indices]
    
    logger.info(f"Created eval split: {len(train_texts)} train, {len(eval_texts)} eval samples")
    
    return train_texts, eval_texts


# =============================================================================
# Trainer
# =============================================================================

class SurrogateTrainer:
    """
    Trainer class implementing surrogate-assisted language model training.
    """
    
    def __init__(
        self,
        config: Config,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        surrogate_model: Optional[PreTrainedModel] = None,
        surrogate_tokenizer: Optional[PreTrainedTokenizer] = None,
        tpu_rank: Optional[int] = None,  # For TPU multi-core training
    ):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.surrogate_model = surrogate_model
        self.surrogate_tokenizer = surrogate_tokenizer
        self.tpu_rank = tpu_rank
        
        # Setup device (TPU, GPU, MPS, or CPU)
        self.device = self._setup_device()
        self.is_tpu = self.device.type == 'xla' if HAS_XLA else False
        
        # Setup distributed training (GPU DDP or TPU)
        self.is_distributed = config.training.local_rank != -1 or (self.is_tpu and config.training.tpu_cores > 1)
        self.world_size = 1
        self.global_rank = 0
        self.local_rank = 0
        # TPU-specific attributes (defaults for single-core, updated by _setup_tpu_distributed)
        self.host_rank = 0
        self.num_hosts = 1
        self.is_master = True
        self.is_host_master = True
        
        if self.is_tpu and config.training.tpu_cores > 1:
            self._setup_tpu_distributed()
        elif config.training.local_rank != -1:
            self._setup_distributed()
        
        # Move models to device
        logger.info(f"Moving models to device: {self.device}")
        self.model = self.model.to(self.device)
        if self.surrogate_model is not None:
            self._validate_and_move_surrogate_to_device()
            self.surrogate_model.eval()
        
        # Setup vocabulary alignment
        self.vocab_aligner = None
        if self.surrogate_model is not None and self.surrogate_tokenizer is not None:
            logger.info("Building vocabulary alignment...")
            self.vocab_aligner = VocabularyAligner(
                self.tokenizer,
                self.surrogate_tokenizer,
                self.device,
            )
            logger.info("Vocabulary alignment complete.")
        
        # Setup data loaders
        logger.info("Setting up data loaders...")
        self._setup_dataloaders()
        logger.info("Data loaders ready.")
        
        # Setup optimizer and scheduler
        logger.info("Setting up optimizer and scheduler...")
        self._setup_optimizer()
        
        # Setup mixed precision (not used for TPU - TPU handles precision internally)
        self._setup_mixed_precision()
        logger.info("Trainer initialization complete.")
        
        # Distributed model wrapper (GPU DDP only, not needed for TPU)
        if config.training.local_rank != -1 and not self.is_tpu:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_eval_loss = float("inf")
        
        # Speed and FLOPS metrics
        self.speed_metrics = SpeedMetrics()
        self.flops_per_token, self.flops_per_batch = estimate_model_flops(
            self.model.module if hasattr(self.model, 'module') else self.model,
            config.training.per_device_train_batch_size,
            config.data.max_seq_length,
        )
        logger.info(f"Estimated FLOPs per token: {self.flops_per_token:,}")
        logger.info(f"Estimated FLOPs per batch: {self.flops_per_batch:,}")
        
        # Setup logging
        self._setup_logging()
        
    def _setup_device(self) -> torch.device:
        """Setup compute device based on configuration and availability."""
        device_type = self.config.training.device.lower()
        
        if device_type == "tpu":
            if not HAS_XLA:
                raise RuntimeError("PyTorch XLA is required for TPU. Install with: pip install torch-xla")
            device = xm.xla_device()
            logger.info(f"Using TPU device: {device}")
            return device
        
        elif device_type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA requested but not available")
            if self.config.training.local_rank != -1:
                return torch.device("cuda", self.config.training.local_rank)
            return torch.device("cuda")
        
        elif device_type == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError("MPS requested but not available")
            return torch.device("mps")
        
        elif device_type == "cpu":
            return torch.device("cpu")
        
        elif device_type == "auto":
            # Auto-detect best available device
            if HAS_XLA:
                try:
                    # Check if we're actually running on TPU
                    device = xm.xla_device()
                    # This will fail if not on TPU
                    _ = torch.zeros(1, device=device)
                    logger.info(f"Auto-detected TPU device: {device}")
                    return device
                except Exception:
                    pass
            
            if torch.cuda.is_available():
                if self.config.training.local_rank != -1:
                    return torch.device("cuda", self.config.training.local_rank)
                return torch.device("cuda")
            elif torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        
        else:
            raise ValueError(f"Unknown device type: {device_type}")

    def _validate_and_move_surrogate_to_device(self) -> None:
        """Validate surrogate model compatibility and move to device.

        For TPU, this performs additional checks:
        - Validates model dtype (TPU doesn't support fp16, only bf16/fp32)
        - Converts model to bf16 if configured for mixed precision
        - Logs warnings for potential compatibility issues
        """
        if self.surrogate_model is None:
            return

        if self.is_tpu:
            # Check current model dtype
            model_dtype = next(self.surrogate_model.parameters()).dtype

            # TPU doesn't support fp16, only bf16 and fp32
            if model_dtype == torch.float16:
                logger.warning(
                    "Surrogate model is in float16, which is NOT supported on TPU. "
                    "Converting to bfloat16 for TPU compatibility."
                )
                self.surrogate_model = self.surrogate_model.to(torch.bfloat16)
            elif model_dtype == torch.bfloat16:
                logger.info("Surrogate model is in bfloat16 (native TPU support)")
            else:
                # float32 - check if we should convert to bf16 for memory efficiency
                if self.config.training.mixed_precision == "bf16":
                    logger.info("Converting surrogate model to bfloat16 for TPU mixed precision")
                    self.surrogate_model = self.surrogate_model.to(torch.bfloat16)
                else:
                    logger.info("Surrogate model is in float32 (TPU will optimize via XLA)")

            # Move to TPU device
            try:
                self.surrogate_model = self.surrogate_model.to(self.device)
                # Test with a small forward pass to catch XLA compilation issues early
                logger.info("Validating surrogate model on TPU...")
                with torch.no_grad():
                    test_input = torch.zeros(1, 8, dtype=torch.long, device=self.device)
                    _ = self.surrogate_model(test_input)
                    if HAS_XLA:
                        xm.mark_step()  # Execute the test computation
                logger.info("Surrogate model validated successfully on TPU")
            except Exception as e:
                logger.error(f"Failed to move/validate surrogate model on TPU: {e}")
                logger.warning(
                    "Surrogate model may not be compatible with TPU. "
                    "Consider using a different surrogate model or disabling surrogate guidance."
                )
                raise RuntimeError(f"Surrogate model TPU validation failed: {e}")
        else:
            # Non-TPU: simple move to device
            self.surrogate_model = self.surrogate_model.to(self.device)

    def _setup_tpu_distributed(self) -> None:
        """Setup distributed training for TPU (single-host and multi-host/pod).
        
        For single-host multi-core (e.g., TPU v3-8):
            - Uses xmp.spawn to create processes
            - world_size = number of cores (e.g., 8)
            
        For multi-host TPU pods (e.g., TPU v3-32, v4-64):
            - Each host runs this script independently
            - PJRT runtime handles cross-host communication
            - world_size = total cores across all hosts
            - Requires proper TPU pod environment setup
        """
        if not HAS_XLA:
            raise RuntimeError("PyTorch XLA required for TPU distributed training")
        
        # Get distributed info from XLA
        self.world_size = xm.xrt_world_size()
        self.global_rank = xm.get_ordinal()
        self.local_rank = xm.get_local_ordinal()
        
        # Calculate host information for TPU pods
        cores_per_host = self.config.training.tpu_cores
        if cores_per_host > 0:
            self.host_rank = self.global_rank // cores_per_host
            self.num_hosts = (self.world_size + cores_per_host - 1) // cores_per_host
        else:
            self.host_rank = 0
            self.num_hosts = 1
        
        # Check if we're the master process (for logging and checkpointing)
        self.is_master = (self.global_rank == 0)
        self.is_host_master = (self.local_rank == 0)  # Master on this host
        
        if self.is_master:
            logger.info(f"TPU distributed training initialized:")
            logger.info(f"  - World size: {self.world_size} cores")
            logger.info(f"  - Number of hosts: {self.num_hosts}")
            logger.info(f"  - Cores per host: {cores_per_host}")
        
        logger.info(f"TPU rank info: global_rank={self.global_rank}/{self.world_size}, "
                    f"local_rank={self.local_rank}, host_rank={self.host_rank}/{self.num_hosts}")
    
    def _setup_distributed(self) -> None:
        """Setup distributed training for GPU (DDP).
        
        Works with both manual DDP setup and torchrun.
        If torchrun is used, the process group is already initialized.
        """
        # Check if process group is already initialized (e.g., by torchrun)
        if not dist.is_initialized():
            # Manual DDP setup
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank = self.config.training.local_rank
        
        if torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
        
        logger.info(f"GPU distributed training: rank {self.global_rank}/{self.world_size}, local_rank={self.local_rank}")
    
    def _setup_dataloaders(self) -> None:
        """Setup training and evaluation data loaders."""
        self.train_sampler = None  # Store reference for set_epoch() calls

        # For TPU distributed training
        if self.is_tpu and self.config.training.tpu_cores > 1:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=True,
            )
        # For GPU distributed training
        elif self.is_distributed and not self.is_tpu:
            self.train_sampler = DistributedSampler(
                self.train_dataset,
                num_replicas=self.world_size,
                rank=self.global_rank,
                shuffle=True,
            )

        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id

        # Don't use pin_memory for TPU
        pin_memory = not self.is_tpu

        # Configure num_workers for data loading
        num_workers = self.config.data.preprocessing_num_workers
        persistent_workers = False
        prefetch_factor = 2  # Default

        if self.is_tpu:
            # TPU with XLA has limited compatibility with multi-worker data loading.
            # We allow 1-2 workers with persistent_workers=True to improve throughput
            # while avoiding XLA graph compilation issues.
            # Key: Use persistent_workers to avoid repeated process spawning, and
            # keep prefetch_factor low to avoid memory bloat.
            if num_workers > 2:
                num_workers = 2
                logger.info(f"TPU detected: Limiting num_workers to {num_workers} for XLA compatibility")
            if num_workers > 0:
                persistent_workers = True
                prefetch_factor = 2
                logger.info(f"TPU data loading: num_workers={num_workers}, persistent_workers=True, prefetch_factor={prefetch_factor}")
            else:
                logger.info("TPU data loading: num_workers=0 (single-threaded)")
        
        collate = partial(
            collate_fn,
            pad_token_id=pad_token_id,
            max_length=self.config.data.max_seq_length,
        )

        # Build DataLoader kwargs conditionally to support older PyTorch versions
        train_loader_kwargs = {
            "batch_size": self.config.training.per_device_train_batch_size,
            "sampler": self.train_sampler,
            "shuffle": (self.train_sampler is None),
            "collate_fn": collate,
            "num_workers": num_workers,
            "pin_memory": pin_memory,
            "drop_last": True,
        }
        if num_workers > 0:
            train_loader_kwargs["persistent_workers"] = persistent_workers
            train_loader_kwargs["prefetch_factor"] = prefetch_factor

        self.train_loader = DataLoader(self.train_dataset, **train_loader_kwargs)
        
        # Wrap with TPU parallel loader for efficient data transfer
        if self.is_tpu and HAS_XLA:
            self.train_loader = pl.MpDeviceLoader(self.train_loader, self.device)
        
        self.eval_loader = None
        if self.eval_dataset is not None:
            eval_loader_kwargs = {
                "batch_size": self.config.training.per_device_eval_batch_size,
                "shuffle": False,
                "collate_fn": collate,
                "num_workers": num_workers,
                "pin_memory": pin_memory,
            }
            if num_workers > 0:
                eval_loader_kwargs["persistent_workers"] = persistent_workers
                eval_loader_kwargs["prefetch_factor"] = prefetch_factor

            eval_loader = DataLoader(self.eval_dataset, **eval_loader_kwargs)
            # Wrap eval loader for TPU
            if self.is_tpu and HAS_XLA:
                self.eval_loader = pl.MpDeviceLoader(eval_loader, self.device)
            else:
                self.eval_loader = eval_loader
    
    def _setup_optimizer(self) -> None:
        """Setup optimizer and learning rate scheduler."""
        # Separate parameters with and without weight decay
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.training.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.training.learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
        )
        
        # Calculate total training steps
        num_update_steps_per_epoch = len(self.train_loader) // self.config.training.gradient_accumulation_steps
        self.total_training_steps = num_update_steps_per_epoch * self.config.training.num_epochs
        
        # Warmup steps
        if self.config.training.warmup_steps is not None:
            warmup_steps = self.config.training.warmup_steps
        else:
            warmup_steps = int(self.total_training_steps * self.config.training.warmup_ratio)
        
        self.scheduler = get_scheduler(
            name=self.config.training.lr_scheduler_type,
            optimizer=self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=self.total_training_steps,
        )
        
        logger.info(f"Total training steps: {self.total_training_steps}, Warmup steps: {warmup_steps}")
    
    def _setup_mixed_precision(self) -> None:
        """Setup mixed precision training."""
        self.scaler = None
        self.autocast_dtype = torch.float32
        self.use_autocast = False
        
        # TPU handles precision internally via XLA, no GradScaler needed
        if self.is_tpu:
            if self.config.training.mixed_precision == "bf16":
                # TPU v3+ supports bfloat16 natively
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
                logger.info("TPU: Using bfloat16 precision")
            else:
                # TPU defaults to float32, XLA optimizes internally
                logger.info("TPU: Using float32 precision (XLA will optimize)")
            return
        
        # CUDA mixed precision
        if self.device.type == "cuda":
            if self.config.training.mixed_precision == "fp16" and HAS_CUDA_AMP:
                self.scaler = GradScaler()
                self.autocast_dtype = torch.float16
                self.use_autocast = True
            elif self.config.training.mixed_precision == "bf16":
                self.autocast_dtype = torch.bfloat16
                self.use_autocast = True
        
        # MPS mixed precision (limited support)
        elif self.device.type == "mps":
            if self.config.training.mixed_precision == "fp16":
                self.autocast_dtype = torch.float16
                self.use_autocast = True

        # Cache TPU autocast availability (check once during setup)
        self._tpu_autocast_available = False
        if self.is_tpu and self.autocast_dtype == torch.bfloat16:
            try:
                # Test if TPU autocast is available (requires PyTorch XLA 2.0+)
                with torch.autocast('xla', dtype=torch.bfloat16, enabled=True):
                    pass
                self._tpu_autocast_available = True
                logger.info("TPU autocast (bfloat16) is available")
            except (ValueError, RuntimeError, TypeError):
                logger.warning(
                    "TPU autocast not available (requires PyTorch XLA 2.0+). "
                    "Training will use the model's native dtype. For bfloat16, "
                    "ensure model was loaded with torch_dtype=torch.bfloat16."
                )

    def _get_autocast_context(self):
        """Get the appropriate autocast context manager for the current device.

        Returns a context manager that enables automatic mixed precision when supported.
        For TPU, this uses torch.autocast('xla', ...) when available (XLA 2.0+).
        Falls back to nullcontext when autocast is not available or not configured.
        """
        from contextlib import nullcontext

        if not self.use_autocast:
            return nullcontext()

        if self.is_tpu:
            if self._tpu_autocast_available:
                return torch.autocast('xla', dtype=self.autocast_dtype, enabled=True)
            else:
                # TPU without autocast support - use nullcontext
                # Model should already be in the right dtype from _validate_and_move_surrogate_to_device
                return nullcontext()

        # CUDA/MPS autocast
        try:
            # PyTorch 2.0+ API
            return torch.autocast(device_type=self.device.type, dtype=self.autocast_dtype)
        except TypeError:
            # Older PyTorch API (torch.cuda.amp.autocast)
            if self.device.type == "cuda" and HAS_CUDA_AMP:
                return autocast(enabled=True, dtype=self.autocast_dtype)
            return nullcontext()

    def _setup_logging(self) -> None:
        """Setup logging with Weights & Biases."""
        if self.global_rank == 0 and HAS_WANDB and self.config.training.wandb_project:
            wandb.init(
                project=self.config.training.wandb_project,
                name=self.config.training.wandb_run_name,
                entity=self.config.training.wandb_entity,
                config={
                    "model": vars(self.config.model),
                    "surrogate": vars(self.config.surrogate),
                    "data": vars(self.config.data),
                    "training": vars(self.config.training),
                },
            )
    
    def get_labels(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Create labels from input_ids (shifted left by 1)."""
        labels = batch["input_ids"].clone()
        
        # Mask padding tokens
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        
        # Shift for next-token prediction
        labels = labels[..., 1:].contiguous()
        
        return labels
    
    def get_surrogate_loss_weight(self) -> float:
        """
        Compute the current surrogate loss weight using cosine decay (no warmup).
        
        The weight decays from loss_weight_initial to loss_weight_final over
        the course of training following a cosine schedule:
        
        weight = final + 0.5 * (initial - final) * (1 + cos(pi * step / total_steps))
        
        Returns:
            Current surrogate loss weight
        """
        if not self.config.surrogate.enabled:
            return 0.0
        
        initial = self.config.surrogate.loss_weight_initial
        final = self.config.surrogate.loss_weight_final
        
        if self.total_training_steps == 0:
            return initial
        
        # Cosine decay without warmup
        progress = min(self.global_step / self.total_training_steps, 1.0)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        
        weight = final + (initial - final) * cosine_decay
        return weight
    
    def compute_surrogate_guidance(
        self,
        batch: Dict[str, torch.Tensor],
        labels: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute top-k tokens from surrogate model selected by probability.
        
        Selection process:
        1. Select top-k tokens by HIGHEST PROBABILITY from surrogate
        2. Gather the PERPLEXITY values for those tokens (used for loss weighting)
        3. Apply probability threshold mask (tokens below threshold get perp=inf)
        
        The perplexity weighting in the loss will then:
        - Give higher weight to tokens with lower perplexity (higher confidence)
        - Zero out tokens masked by the probability threshold (perp=inf -> weight=0)
        
        Returns:
            Tuple of (perp_values, perp_indices) or (None, None) if surrogate is disabled
            - perp_values: (batch_size, seq_len-1, k) perplexity values for weighting
            - perp_indices: (batch_size, seq_len-1, k) token indices in surrogate vocab
        """
        if self.surrogate_model is None or not self.config.surrogate.enabled:
            return None, None
        
        k = self.config.surrogate.k
        prob_threshold = self.config.surrogate.probability_threshold
        
        with torch.no_grad():
            input_ids = batch["input_ids"]  # (batch_size, seq_len)
            attention_mask = batch.get("attention_mask")
            batch_size, seq_len = input_ids.shape
            
            # === DIRECT GPU TOKEN TRANSLATION (no decode/re-encode) ===
            # Translate base model token IDs to surrogate token IDs directly
            surrogate_input_ids = self.vocab_aligner.lookup_base_to_surrogate[input_ids]
            
            # Create mask for untranslatable tokens (those not in vocabulary intersection)
            untranslatable_mask = (surrogate_input_ids == -100)
            
            # Replace untranslatable tokens with surrogate's pad token for forward pass
            surrogate_pad_id = self.surrogate_tokenizer.pad_token_id
            if surrogate_pad_id is None:
                surrogate_pad_id = self.surrogate_tokenizer.eos_token_id
            
            surrogate_input_ids_safe = surrogate_input_ids.clone()
            surrogate_input_ids_safe[untranslatable_mask] = surrogate_pad_id
            
            # Create attention mask: mask out untranslatable tokens
            if attention_mask is not None:
                surrogate_attention_mask = attention_mask.clone()
                surrogate_attention_mask[untranslatable_mask] = 0
            else:
                surrogate_attention_mask = (~untranslatable_mask).long()
            
            # Forward pass through surrogate model (all on GPU)
            surrogate_outputs = self.surrogate_model(
                input_ids=surrogate_input_ids_safe,
                attention_mask=surrogate_attention_mask,
            )
            # Shift logits to align with labels (next-token prediction)
            # surrogate_logits: (batch_size, seq_len-1, vocab_size)
            surrogate_logits = surrogate_outputs.logits[..., :-1, :].contiguous()

            # TPU: Mark step after surrogate forward to execute and free memory
            # This prevents XLA graph from accumulating the entire surrogate computation
            if self.is_tpu and HAS_XLA:
                xm.mark_step()

            surr_vocab_size = surrogate_logits.shape[-1]
            label_seq_len = labels.shape[1]  # seq_len - 1 due to shift in get_labels
            
            # Compute probability distribution
            surrogate_probs = F.softmax(surrogate_logits, dim=-1)
            # Shape: (batch_size, seq_len-1, surr_vocab_size)
            
            # === MASKING (applied to probabilities for selection) === 
            # We mask by setting probability to 0 (so they won't be selected in top-k)
            
            # 1. Mask out tokens not in vocabulary intersection (can't translate back)
            vocab_mask = ~torch.isin(
                torch.arange(surr_vocab_size, device=self.device),
                self.vocab_aligner.surrogate_permitted_ids,
            )
            surrogate_probs[:, :, vocab_mask] = 0.0
            
            # 2. Mask positions corresponding to labels=-100 (padding/ignored)
            invalid_label_positions = (labels == -100).unsqueeze(-1)
            surrogate_probs = surrogate_probs.masked_fill(invalid_label_positions, 0.0)
            
            # 3. Mask positions where the INPUT token was untranslatable
            #    (shifted by 1 to align with logits/labels which predict next token)
            untranslatable_positions = untranslatable_mask[:, 1:].unsqueeze(-1)
            surrogate_probs = surrogate_probs.masked_fill(untranslatable_positions, 0.0)
            
            # 4. Mask out the actual target token (we don't want it in top-k)
            translated_labels = self.vocab_aligner.translate_base_to_surrogate(labels)
            valid_label_mask = translated_labels != -100
            
            batch_indices = torch.arange(batch_size, device=self.device).unsqueeze(1).expand(-1, label_seq_len)
            seq_indices = torch.arange(label_seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
            
            # Use 0 for invalid labels to avoid indexing errors (they're already masked anyway)
            safe_translated_labels = translated_labels.clone()
            safe_translated_labels[~valid_label_mask] = 0
            surrogate_probs[batch_indices, seq_indices, safe_translated_labels] = 0.0

            # TPU: Mark step after masking operations to execute accumulated computations
            if self.is_tpu and HAS_XLA:
                xm.mark_step()

            # === TOP-K SELECTION BY PROBABILITY (fully on GPU) ===
            # Select k tokens with HIGHEST probability from surrogate
            topk_result = torch.topk(surrogate_probs, k=k, largest=True, sorted=True, dim=-1)
            prob_values = topk_result.values   # (batch_size, seq_len-1, k)
            perp_indices = topk_result.indices  # (batch_size, seq_len-1, k) - in surrogate vocab
            
            # === PERPLEXITY VALUES FOR SELECTED TOKENS (OPTIONAL) ===
            # If perplexity weighting is enabled, compute perplexity in float32 for stability.
            # Otherwise, skip the second softmax and create a zero tensor that only carries
            # validity masks (inf for invalid tokens).
            if self.config.surrogate.use_perplexity_weighting:
                # Compute perplexity for the full vocab, then gather for selected indices
                # perplexity = 1 / probability
                # CRITICAL: Compute in float32 to avoid overflow/underflow in float16/MPS
                surrogate_probs_f32 = F.softmax(surrogate_logits.float(), dim=-1)
                surrogate_perp_full = torch.reciprocal(surrogate_probs_f32 + 1e-6)
                perp_values = torch.gather(surrogate_perp_full, dim=2, index=perp_indices)
                # Clamp to avoid extreme values (max perp of 1e6 = prob of 1e-6)
                perp_values = perp_values.clamp(max=1e6)
            else:
                # Placeholder values; will be masked with inf where invalid
                perp_values = torch.zeros_like(prob_values, dtype=torch.float32)
            # Shape: (batch_size, seq_len-1, k)
            
            # === PROBABILITY THRESHOLD FILTERING ===
            # Mask out tokens below the probability threshold by setting their perplexity to inf
            # This will cause them to get zero weight in the softmax(-perplexity) weighting
            if prob_threshold > 0:
                below_threshold_mask = prob_values < prob_threshold
                perp_values = perp_values.masked_fill(below_threshold_mask, float('inf'))
            
            # Also mask tokens that had zero probability (from our earlier masking)
            # These would have perp = 1/0 = inf anyway, but let's be explicit
            zero_prob_mask = prob_values == 0.0
            perp_values = perp_values.masked_fill(zero_prob_mask, float('inf'))
            
            # === CRITICAL: Zero out entire row when TARGET token doesn't exist in intersection ===
            # If the target token can't be translated to surrogate vocab, we can't trust the
            # surrogate's guidance for this position. Set entire row to inf so the weighting
            # mechanism zeros out the contribution (softmax of -inf -> 0).
            target_not_in_intersection = (translated_labels == -100) & (labels != -100)
            target_not_in_intersection_mask = target_not_in_intersection.unsqueeze(-1)  # (batch, seq, 1)
            perp_values = perp_values.masked_fill(target_not_in_intersection_mask, float('inf'))

            # TPU: Final mark_step to execute all computations before returning
            # This ensures the XLA graph for surrogate guidance is complete and executed
            if self.is_tpu and HAS_XLA:
                xm.mark_step()

        return perp_values, perp_indices
    
    def compute_surrogate_logits(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """
        Compute surrogate model logits for KL divergence loss.
        
        Returns:
            Surrogate logits (batch_size, seq_len-1, vocab_size) or None if surrogate disabled
        """
        if self.surrogate_model is None or not self.config.surrogate.enabled:
            return None
        
        with torch.no_grad():
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            
            # Translate base model token IDs to surrogate token IDs
            surrogate_input_ids = self.vocab_aligner.lookup_base_to_surrogate[input_ids]
            
            # Create mask for untranslatable tokens
            untranslatable_mask = (surrogate_input_ids == -100)
            
            # Replace untranslatable tokens with surrogate's pad token
            surrogate_pad_id = self.surrogate_tokenizer.pad_token_id
            if surrogate_pad_id is None:
                surrogate_pad_id = self.surrogate_tokenizer.eos_token_id
            
            surrogate_input_ids_safe = surrogate_input_ids.clone()
            surrogate_input_ids_safe[untranslatable_mask] = surrogate_pad_id
            
            # Create attention mask for surrogate
            if attention_mask is not None:
                surrogate_attention_mask = attention_mask.clone()
                surrogate_attention_mask[untranslatable_mask] = 0
            else:
                surrogate_attention_mask = (~untranslatable_mask).long()
            
            # Forward pass through surrogate model
            surrogate_outputs = self.surrogate_model(
                input_ids=surrogate_input_ids_safe,
                attention_mask=surrogate_attention_mask,
            )

            # Shift logits to align with labels (next-token prediction)
            surrogate_logits = surrogate_outputs.logits[..., :-1, :].contiguous()

            # TPU: Mark step after surrogate forward to execute and bound XLA graph
            if self.is_tpu and HAS_XLA:
                xm.mark_step()

        return surrogate_logits
    
    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Execute a single training step.
        
        Supports three loss types:
        - "standard": Cross-entropy only (no surrogate)
        - "surrogate": CE + surrogate-guided loss with top-k perplexity weighting
        - "kl": CE + KL divergence from surrogate distribution
        """
        # For TPU with MpDeviceLoader, data is already on device
        if not self.is_tpu:
            batch = {k: v.to(self.device) for k, v in batch.items()}
        
        # Get labels
        labels = self.get_labels(batch)
        
        loss_type = self.config.training.loss_type

        # Forward pass with appropriate autocast context
        with self._get_autocast_context():
            # Forward pass through base model
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch.get("attention_mask"),
            )
            
            # Shift logits for next-token prediction
            logits = outputs.logits[..., :-1, :].contiguous()
            
            # Get current surrogate loss weight (cosine decay)
            surrogate_weight = self.get_surrogate_loss_weight()
            num_aux_tokens = None  # Will be set if using SDCE loss
            
            if loss_type == "standard":
                # Standard cross-entropy only
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                ce_loss, z_loss = standard_cross_entropy_loss(
                    logits=logits_flat,
                    labels=labels_flat,
                    ignore_index=-100,
                    reduction="mean",
                    compute_z_loss=self.config.training.use_z_loss,
                    z_loss_multiplier=self.config.training.z_loss_multiplier,
                )
                
            elif loss_type == "kl":
                # KL divergence from surrogate distribution
                surrogate_logits = self.compute_surrogate_logits(batch)

                if surrogate_logits is not None and surrogate_weight > 0:
                    # Get vocabulary intersection indices for cross-vocab KL
                    student_vocab_indices = None
                    teacher_vocab_indices = None
                    if self.vocab_aligner is not None:
                        student_vocab_indices = self.vocab_aligner.base_permitted_ids
                        teacher_vocab_indices = self.vocab_aligner.surrogate_permitted_ids

                    ce_loss, z_loss = kl_divergence_loss(
                        student_logits=logits,
                        teacher_logits=surrogate_logits,
                        labels=labels,
                        temperature=1.0,
                        kl_weight=surrogate_weight,
                        ignore_index=-100,
                        reduction="mean",
                        compute_z_loss=self.config.training.use_z_loss,
                        z_loss_multiplier=self.config.training.z_loss_multiplier,
                        student_vocab_indices=student_vocab_indices,
                        teacher_vocab_indices=teacher_vocab_indices,
                    )
                else:
                    # Fall back to standard CE if surrogate not available
                    logits_flat = logits.view(-1, logits.size(-1))
                    labels_flat = labels.view(-1)
                    ce_loss, z_loss = standard_cross_entropy_loss(
                        logits=logits_flat,
                        labels=labels_flat,
                        ignore_index=-100,
                        reduction="mean",
                        compute_z_loss=self.config.training.use_z_loss,
                        z_loss_multiplier=self.config.training.z_loss_multiplier,
                    )
                    
            else:  # loss_type == "surrogate" (default)
                # Surrogate-guided loss with top-k perplexity weighting
                perp_values, perp_indices = self.compute_surrogate_guidance(batch, labels)
                
                # Flatten for loss computation
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                lookup_table = None
                if self.vocab_aligner is not None:
                    lookup_table = self.vocab_aligner.lookup_surrogate_to_base
                
                ce_loss, z_loss, num_aux_tokens = surrogate_cross_entropy_loss(
                    logits=logits_flat if perp_indices is None else logits,
                    labels=labels_flat if perp_indices is None else labels,
                    perp_values=perp_values,
                    perp_indices=perp_indices,
                    lookup_surrogate_to_self=lookup_table,
                    surrogate_weight=surrogate_weight,
                    ignore_index=-100,
                    reduction="mean",
                    compute_z_loss=self.config.training.use_z_loss,
                    z_loss_multiplier=self.config.training.z_loss_multiplier,
                    use_perplexity_weighting=self.config.surrogate.use_perplexity_weighting,
                )
            
            loss = ce_loss
            if z_loss is not None:
                loss = loss + z_loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.training.gradient_accumulation_steps
        
        # Backward pass
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        # For TPU with gradient accumulation, mark_step() after each backward
        # prevents XLA graph from growing too large during accumulation
        if self.is_tpu and self.config.training.gradient_accumulation_steps > 1:
            xm.mark_step()

        metrics = {
            "loss": ce_loss.item(),
            "perplexity": math.exp(min(ce_loss.item(), 20)),
            "surrogate_weight": surrogate_weight if loss_type != "standard" else 0.0,
            "loss_type": loss_type,
        }
        if z_loss is not None:
            metrics["z_loss"] = z_loss.item()
        if num_aux_tokens is not None:
            metrics["num_aux_tokens"] = num_aux_tokens
        
        return metrics
    
    def optimizer_step(self) -> Dict[str, float]:
        """Execute optimizer step with gradient clipping."""
        metrics = {}
        
        if self.scaler is not None:
            self.scaler.unscale_(self.optimizer)
        
        # Gradient clipping
        if self.is_tpu:
            # TPU gradient clipping - must be done manually before optimizer step
            # First reduce gradients across TPU cores
            xm.reduce_gradients(self.optimizer)
            # Mark step to execute reduction before accessing gradient values
            xm.mark_step()
            # Clip gradients (now safe to access reduced gradient values)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            # Step optimizer
            self.optimizer.step()
            xm.mark_step()  # Execute optimizer step
            metrics["grad_norm"] = grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.training.max_grad_norm,
            )
            metrics["grad_norm"] = grad_norm.item()
            
            # Optimizer step
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
        
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        metrics["learning_rate"] = self.scheduler.get_last_lr()[0]
        
        return metrics
    
    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Run evaluation loop."""
        if self.eval_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        total_tokens = 0

        for batch in tqdm(self.eval_loader, desc="Evaluating", disable=self.global_rank != 0):
            # For TPU with MpDeviceLoader, data is already on device
            if not self.is_tpu:
                batch = {k: v.to(self.device) for k, v in batch.items()}
            labels = self.get_labels(batch)

            with self._get_autocast_context():
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch.get("attention_mask"),
                )
                
                logits = outputs.logits[..., :-1, :].contiguous()
                logits_flat = logits.view(-1, logits.size(-1))
                labels_flat = labels.view(-1)
                
                loss = F.cross_entropy(logits_flat, labels_flat, ignore_index=-100, reduction="sum")
            
            num_tokens = (labels_flat != -100).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
            
            # Mark step for TPU
            if self.is_tpu:
                xm.mark_step()
        
        # Aggregate across ranks
        if self.is_distributed:
            total_loss_tensor = torch.tensor(total_loss, device=self.device)
            total_tokens_tensor = torch.tensor(total_tokens, device=self.device)

            if self.is_tpu:
                # TPU all-reduce - must call mark_step() before accessing values
                total_loss_tensor = xm.all_reduce(xm.REDUCE_SUM, total_loss_tensor)
                total_tokens_tensor = xm.all_reduce(xm.REDUCE_SUM, total_tokens_tensor)
                xm.mark_step()  # Execute the all-reduce before accessing tensor values
            else:
                # GPU all-reduce
                dist.all_reduce(total_loss_tensor)
                dist.all_reduce(total_tokens_tensor)

            total_loss = total_loss_tensor.item()
            total_tokens = total_tokens_tensor.item()
        
        avg_loss = total_loss / (total_tokens + 1e-8)
        perplexity = math.exp(min(avg_loss, 20))
        
        self.model.train()
        
        return {
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
        }
    
    def run_benchmarks(self) -> Dict[str, float]:
        """
        Run lm-evaluation-harness benchmarks and return scores.
        
        Note: lm-evaluation-harness does not support TPU. On TPU, this method
        saves a checkpoint for offline evaluation instead.
        
        Returns:
            Dictionary of benchmark scores
        """
        if not self.config.evaluation.enabled:
            return {}
        
        if self.global_rank != 0:
            # Only run benchmarks on main process
            return {}
        
        # lm-evaluation-harness does not support TPU/XLA devices
        # Save a checkpoint for offline evaluation instead
        if self.is_tpu:
            benchmark_checkpoint_path = os.path.join(
                self.config.training.output_dir,
                f"benchmark-checkpoint-step-{self.global_step}"
            )
            logger.info(
                f"lm-evaluation-harness does not support TPU. "
                f"Saving checkpoint for offline evaluation: {benchmark_checkpoint_path}"
            )
            self.save_checkpoint(benchmark_checkpoint_path)
            return {}
        
        if not HAS_LM_EVAL:
            logger.warning(
                "lm-evaluation-harness not installed. "
                "Install with: pip install lm-eval"
            )
            return {}
        
        logger.info(f"Running benchmarks: {self.config.evaluation.tasks}")
        
        self.model.eval()
        
        try:
            # Get the underlying model (unwrap DDP if necessary)
            model_to_eval = self.model.module if hasattr(self.model, "module") else self.model
            
            # Create lm-eval model wrapper
            lm_model = HFLM(
                pretrained=model_to_eval,
                tokenizer=self.tokenizer,
                batch_size=self.config.evaluation.batch_size,
                device=str(self.device),
            )
            
            # Run evaluation
            results = simple_evaluate(
                model=lm_model,
                tasks=self.config.evaluation.tasks,
                num_fewshot=self.config.evaluation.num_fewshot,
                limit=self.config.evaluation.limit,
                log_samples=False,
            )
            
            # Extract metrics
            benchmark_metrics = {}
            task_scores = []
            
            for task_name, task_results in results.get("results", {}).items():
                # Get the main accuracy/score metric
                if "acc" in task_results:
                    score = task_results["acc"]
                    metric_name = "acc"
                elif "acc_norm" in task_results:
                    score = task_results["acc_norm"]
                    metric_name = "acc_norm"
                elif "exact_match" in task_results:
                    score = task_results["exact_match"]
                    metric_name = "exact_match"
                else:
                    # Take first numeric metric
                    for k, v in task_results.items():
                        if isinstance(v, (int, float)) and not k.endswith("_stderr"):
                            score = v
                            metric_name = k
                            break
                    else:
                        continue
                
                task_scores.append(score)
                
                if self.config.evaluation.log_individual_tasks:
                    benchmark_metrics[f"benchmark/{task_name}/{metric_name}"] = score
                    
                    # Also log stderr if available
                    stderr_key = f"{metric_name}_stderr"
                    if stderr_key in task_results:
                        benchmark_metrics[f"benchmark/{task_name}/{stderr_key}"] = task_results[stderr_key]
            
            # Compute aggregate score
            if self.config.evaluation.log_aggregate_score and task_scores:
                benchmark_metrics["benchmark/average_score"] = sum(task_scores) / len(task_scores)
            
            logger.info(f"Benchmark results: {benchmark_metrics}")
            
        except Exception as e:
            logger.error(f"Error running benchmarks: {e}")
            benchmark_metrics = {}
        
        self.model.train()
        
        return benchmark_metrics
    
    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        # Synchronize all ranks before checkpoint (ensures consistent state)
        if self.is_tpu and self.is_distributed:
            xm.rendezvous("pre_checkpoint_barrier")
        elif self.is_distributed:
            dist.barrier()

        if self.global_rank != 0:
            # Non-master ranks wait for master to finish saving
            if self.is_tpu and self.is_distributed:
                xm.rendezvous("post_checkpoint_barrier")
            elif self.is_distributed:
                dist.barrier()
            return

        os.makedirs(path, exist_ok=True)

        # Save model
        model_to_save = self.model.module if hasattr(self.model, "module") else self.model
        model_to_save.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # Save training state
        state = {
            "global_step": self.global_step,
            "epoch": self.epoch,
            "best_eval_loss": self.best_eval_loss,
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                "xla": xm.get_rng_state() if (self.is_tpu and HAS_XLA) else None,
            },
        }
        if self.scaler is not None:
            state["scaler_state"] = self.scaler.state_dict()
        
        torch.save(state, os.path.join(path, "training_state.pt"))
        self.config.save(os.path.join(path, "config.json"))

        logger.info(f"Checkpoint saved to {path}")

        # Synchronize after checkpoint save (allow other ranks to proceed)
        if self.is_tpu and self.is_distributed:
            xm.rendezvous("post_checkpoint_barrier")
        elif self.is_distributed:
            dist.barrier()

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        state_path = os.path.join(path, "training_state.pt")
        if not os.path.exists(state_path):
            logger.warning(f"No training state found at {state_path}")
            return
        
        state = torch.load(state_path, map_location=self.device)
        
        self.global_step = state["global_step"]
        self.epoch = state["epoch"]
        self.best_eval_loss = state["best_eval_loss"]
        self.optimizer.load_state_dict(state["optimizer_state"])
        self.scheduler.load_state_dict(state["scheduler_state"])
        
        if self.scaler is not None and "scaler_state" in state:
            self.scaler.load_state_dict(state["scaler_state"])
        
        # Restore RNG states
        rng_state = state["rng_state"]
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and rng_state["cuda"] is not None:
            torch.cuda.set_rng_state(rng_state["cuda"])
        if self.is_tpu and HAS_XLA and rng_state.get("xla") is not None:
            xm.set_rng_state(rng_state["xla"])
        
        logger.info(f"Checkpoint loaded from {path}, resuming from step {self.global_step}")
    
    def train(self) -> None:
        """Main training loop."""
        logger.info("Starting training...")
        logger.info(f"  Num epochs: {self.config.training.num_epochs}")
        logger.info(f"  Batch size per device: {self.config.training.per_device_train_batch_size}")
        logger.info(f"  Gradient accumulation steps: {self.config.training.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps: {self.total_training_steps}")
        logger.info(f"  Surrogate model enabled: {self.config.surrogate.enabled}")
        if self.config.surrogate.enabled:
            logger.info(f"  Surrogate model: {self.config.surrogate.name_or_path}")
            logger.info(f"  Surrogate k: {self.config.surrogate.k}")
        if self.config.evaluation.enabled:
            logger.info(f"  Benchmark tasks: {self.config.evaluation.tasks}")
            logger.info(f"  Benchmark eval interval: {self.config.evaluation.eval_interval}")
        
        # Resume from checkpoint if specified
        if self.config.training.resume_from_checkpoint:
            self.load_checkpoint(self.config.training.resume_from_checkpoint)
        
        self.model.train()
        self.optimizer.zero_grad()
        
        # Initialize speed metrics
        self.speed_metrics.start()
        running_loss = 0.0
        running_grad_norm = 0.0
        running_steps = 0
        
        # Compute batch tokens
        batch_tokens = (
            self.config.training.per_device_train_batch_size 
            * self.config.data.max_seq_length
        )
        global_batch_tokens = batch_tokens * self.world_size * self.config.training.gradient_accumulation_steps
        
        for epoch in range(self.epoch, self.config.training.num_epochs):
            self.epoch = epoch

            # Set epoch on sampler for proper shuffling in distributed training
            # Use stored reference since MpDeviceLoader wraps the sampler
            if self.is_distributed and self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            
            progress_bar = tqdm(
                self.train_loader,
                desc=f"Epoch {epoch}",
                disable=self.global_rank != 0,
            )
            
            for step, batch in enumerate(progress_bar):
                # Training step
                metrics = self.train_step(batch)
                running_loss += metrics["loss"]
                running_steps += 1
                
                # Update speed metrics
                self.speed_metrics.update(batch_tokens)
                
                # Optimizer step (with gradient accumulation)
                if (step + 1) % self.config.training.gradient_accumulation_steps == 0:
                    optim_metrics = self.optimizer_step()
                    self.global_step += 1
                    running_grad_norm += optim_metrics["grad_norm"]
                    
                    # Logging
                    if self.global_step % self.config.training.logging_steps == 0:
                        avg_loss = running_loss / running_steps
                        avg_grad_norm = running_grad_norm / (self.global_step % self.config.training.logging_steps or self.config.training.logging_steps)
                        
                        # Get speed metrics
                        speed = self.speed_metrics.get_metrics(self.world_size)
                        
                        # Compute FLOPS metrics
                        flops_per_second = speed["tokens_per_second_global"] * self.flops_per_token
                        flops_per_second_per_device = speed["tokens_per_second"] * self.flops_per_token
                        total_flops = self.speed_metrics.total_tokens * self.world_size * self.flops_per_token
                        
                        log_metrics = {
                            # Training metrics
                            "train/loss": avg_loss,
                            "train/perplexity": math.exp(min(avg_loss, 20)),
                            "train/learning_rate": optim_metrics["learning_rate"],
                            "train/grad_norm": avg_grad_norm,
                            "train/surrogate_weight": metrics.get("surrogate_weight", 0.0),
                            
                            # Throughput metrics
                            "throughput/tokens_per_second": speed["tokens_per_second_global"],
                            "throughput/tokens_per_second_per_device": speed["tokens_per_second"],
                            "throughput/batches_per_second": speed["batches_per_second"],
                            "throughput/total_tokens_trained": speed["total_tokens_trained"],
                            "throughput/total_tokens_per_second": speed["total_tokens_per_second"],
                            
                            # FLOPS metrics
                            "flops/total_tflops": total_flops / 1e12,
                            "flops/tflops_per_second": flops_per_second / 1e12,
                            "flops/tflops_per_second_per_device": flops_per_second_per_device / 1e12,
                            
                            # Progress
                            "progress/global_step": self.global_step,
                            "progress/epoch": epoch,
                            "progress/percent_complete": (self.global_step / self.total_training_steps) * 100,
                        }
                        
                        # Add SDCE-specific metrics if available
                        if metrics.get("num_aux_tokens") is not None:
                            log_metrics["train/num_aux_tokens"] = metrics["num_aux_tokens"]
                        
                        if self.global_rank == 0:
                            progress_bar.set_postfix(
                                loss=f"{avg_loss:.4f}", 
                                ppl=f"{math.exp(min(avg_loss, 20)):.2f}",
                                tok_s=f"{speed['tokens_per_second_global']:.0f}"
                            )
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(log_metrics, step=self.global_step)
                        
                        # Reset running metrics
                        running_loss = 0.0
                        running_grad_norm = 0.0
                        running_steps = 0
                        self.speed_metrics.reset()
                    
                    # Evaluation (loss-based)
                    if self.global_step % self.config.training.eval_steps == 0:
                        eval_metrics = self.evaluate()
                        
                        if eval_metrics and self.global_rank == 0:
                            logger.info(f"Step {self.global_step}: {eval_metrics}")
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(eval_metrics, step=self.global_step)
                            
                            # Save best model
                            if eval_metrics["eval_loss"] < self.best_eval_loss:
                                self.best_eval_loss = eval_metrics["eval_loss"]
                                best_path = os.path.join(self.config.training.output_dir, "best_model")
                                self.save_checkpoint(best_path)
                    
                    # Benchmark evaluation (lm-eval-harness)
                    if (self.config.evaluation.enabled and 
                        self.global_step % self.config.evaluation.eval_interval == 0):
                        benchmark_metrics = self.run_benchmarks()
                        
                        if benchmark_metrics and self.global_rank == 0:
                            logger.info(f"Step {self.global_step} benchmarks: {benchmark_metrics}")
                            
                            if HAS_WANDB and wandb.run is not None:
                                wandb.log(benchmark_metrics, step=self.global_step)
                    
                    # Save checkpoint
                    if self.global_step % self.config.training.save_steps == 0:
                        checkpoint_path = os.path.join(
                            self.config.training.output_dir,
                            f"checkpoint-{self.global_step}",
                        )
                        self.save_checkpoint(checkpoint_path)
                        
                        # Remove old checkpoints
                        self._cleanup_checkpoints()
                    
                    # Check if training is complete
                    if self.global_step >= self.total_training_steps:
                        break
            
            if self.global_step >= self.total_training_steps:
                break
        
        # Final evaluation
        logger.info("Running final evaluation...")
        eval_metrics = self.evaluate()
        if eval_metrics and self.global_rank == 0:
            logger.info(f"Final evaluation: {eval_metrics}")
            if HAS_WANDB and wandb.run is not None:
                wandb.log(eval_metrics, step=self.global_step)
        
        # Final benchmarks
        if self.config.evaluation.enabled:
            logger.info("Running final benchmarks...")
            benchmark_metrics = self.run_benchmarks()
            if benchmark_metrics and self.global_rank == 0:
                logger.info(f"Final benchmarks: {benchmark_metrics}")
                if HAS_WANDB and wandb.run is not None:
                    wandb.log(benchmark_metrics, step=self.global_step)
        
        final_path = os.path.join(self.config.training.output_dir, "final_model")
        self.save_checkpoint(final_path)
        
        logger.info("Training complete!")
    
    def _cleanup_checkpoints(self) -> None:
        """Remove old checkpoints to stay within save_total_limit."""
        if self.global_rank != 0 or self.config.training.save_total_limit <= 0:
            return
        
        checkpoint_dirs = []
        for name in os.listdir(self.config.training.output_dir):
            if name.startswith("checkpoint-"):
                path = os.path.join(self.config.training.output_dir, name)
                if os.path.isdir(path):
                    step = int(name.split("-")[1])
                    checkpoint_dirs.append((step, path))
        
        checkpoint_dirs.sort(key=lambda x: x[0])
        
        while len(checkpoint_dirs) > self.config.training.save_total_limit:
            _, oldest_path = checkpoint_dirs.pop(0)
            logger.info(f"Removing old checkpoint: {oldest_path}")
            import shutil
            shutil.rmtree(oldest_path)
    
    def close(self) -> None:
        """Cleanup resources."""
        # GPU distributed cleanup
        if self.is_distributed and not self.is_tpu:
            dist.destroy_process_group()
        
        # TPU cleanup - rendezvous to ensure all cores finish
        if self.is_tpu and HAS_XLA:
            xm.rendezvous("training_complete")
        
        if HAS_WANDB and wandb.run is not None and self.global_rank == 0:
            wandb.finish()


# =============================================================================
# Main
# =============================================================================

def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Surrogate-Assisted Language Model Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Config file
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    
    # Model arguments
    parser.add_argument("--base_model", type=str, help="Base model name or path")
    parser.add_argument("--model_dtype", type=str, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--init_from_scratch", action="store_true",
                        help="Initialize model with random weights instead of pretrained weights")
    
    # Surrogate arguments
    parser.add_argument("--surrogate_model", type=str, help="Surrogate model name or path")
    parser.add_argument("--surrogate_k", type=int, help="Number of top-k tokens to select by probability")
    parser.add_argument("--probability_threshold", type=float, default=None,
                        help="Min probability for surrogate tokens (e.g., 0.02). Tokens below this are masked out.")
    parser.add_argument("--surrogate_dtype", type=str, choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--no_surrogate", action="store_true", help="Disable surrogate model")
    parser.add_argument("--surrogate_loss_weight_initial", type=float, help="Initial surrogate loss weight")
    parser.add_argument("--surrogate_loss_weight_final", type=float, help="Final surrogate loss weight after decay")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, help="HuggingFace dataset name")
    parser.add_argument("--dataset_config", type=str, help="Dataset configuration")
    parser.add_argument("--train_file", type=str, help="Path to custom training file")
    parser.add_argument("--eval_file", type=str, help="Path to custom evaluation file")
    parser.add_argument("--text_column", type=str, help="Column name containing text")
    parser.add_argument("--max_seq_length", type=int, help="Maximum sequence length")
    parser.add_argument("--eval_split_ratio", type=float, default=0.05,
                        help="Ratio of train data to use for eval if no eval split exists")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--num_epochs", type=int, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_ratio", type=float)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mixed_precision", type=str, choices=["fp16", "bf16", "no"])
    parser.add_argument("--use_z_loss", action="store_true")
    
    # Loss type selection
    parser.add_argument("--loss_type", type=str, choices=["standard", "surrogate", "kl"],
                        default=None,
                        help="Loss type: 'standard' (CE only), 'surrogate' (CE + top-k guided), 'kl' (CE + KL divergence)")
    parser.add_argument("--standard_training", action="store_true",
                        help="Use standard cross-entropy training (equivalent to --loss_type standard --no_surrogate)")
    parser.add_argument("--kl_divergence", action="store_true",
                        help="Use KL divergence loss from surrogate (equivalent to --loss_type kl)")
    
    # Logging arguments
    parser.add_argument("--logging_steps", type=int)
    parser.add_argument("--eval_steps", type=int)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_run_name", type=str)
    
    # Checkpoint arguments
    parser.add_argument("--resume_from_checkpoint", type=str)
    
    # Distributed training (for torchrun compatibility)
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training (set automatically by torchrun)")
    
    # Device selection
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "mps", "tpu", "cpu"],
                        default="auto", help="Device to use for training")
    parser.add_argument("--tpu_cores", type=int, default=1, 
                        help="Number of TPU cores per host (1 for single, 8 for v3-8)")
    parser.add_argument("--tpu_num_hosts", type=int, default=1,
                        help="Number of TPU hosts for TPU Pod training (1 for single host, >1 for pods)")
    
    return parser.parse_args()


def _init_model_from_config(
    model_config: ModelConfig,
    for_training: bool = True,
) -> PreTrainedModel:
    """
    Initialize a model based on configuration.
    
    Args:
        model_config: Model configuration
        for_training: If True, apply training-specific settings like gradient checkpointing
        
    Returns:
        Initialized model (either pretrained or from scratch)
    """
    if model_config.init_from_scratch:
        # Initialize model from scratch with random weights
        logger.info(f"Initializing model from scratch using config: {model_config.name_or_path}")
        config = AutoConfig.from_pretrained(
            model_config.name_or_path,
            trust_remote_code=model_config.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=model_config.get_torch_dtype(),
            trust_remote_code=model_config.trust_remote_code,
        )
        
        # Initialize weights properly
        def _init_weights(module):
            """Initialize weights for transformer modules."""
            if hasattr(module, 'weight') and module.weight is not None:
                if module.weight.dim() >= 2:
                    # Use Xavier/Glorot initialization for weight matrices
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    # Use normal initialization for 1D weights (biases, layernorm)
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        
        model.apply(_init_weights)
        logger.info(f"Model initialized from scratch with {sum(p.numel() for p in model.parameters()):,} parameters")
    else:
        # Load pretrained weights
        logger.info(f"Loading pretrained model: {model_config.name_or_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_config.name_or_path,
            torch_dtype=model_config.get_torch_dtype(),
            trust_remote_code=model_config.trust_remote_code,
        )
    
    if for_training and model_config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    
    return model


def _setup_torchrun_distributed() -> Tuple[int, int, int]:
    """
    Setup distributed training for torchrun.
    
    Detects torchrun environment variables and initializes the process group.
    
    Returns:
        Tuple of (local_rank, global_rank, world_size)
    """
    # Check for torchrun environment variables
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    global_rank = int(os.environ.get("RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if local_rank == -1:
        # Not running under torchrun
        return -1, 0, 1
    
    # Initialize process group
    if not dist.is_initialized():
        # Use NCCL for GPU, Gloo for CPU
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    
    # Set CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    
    logger.info(f"Torchrun distributed training initialized: local_rank={local_rank}, "
                f"global_rank={global_rank}, world_size={world_size}")
    
    return local_rank, global_rank, world_size


def _train_fn(rank: int, config: Config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset):
    """Training function for TPU multi-core training via xmp.spawn (single host)."""
    # Set seed with rank offset for different data ordering
    set_seed(config.training.seed + rank)
    
    # Initialize base model
    logger.info(f"[Rank {rank}] Initializing base model: {config.model.name_or_path}")
    base_model = _init_model_from_config(config.model, for_training=True)
    
    surrogate_model = None
    if config.surrogate.enabled:
        logger.info(f"[Rank {rank}] Loading surrogate model: {config.surrogate.name_or_path}")
        surrogate_model = AutoModelForCausalLM.from_pretrained(
            config.surrogate.name_or_path,
            torch_dtype=config.surrogate.get_torch_dtype(),
            trust_remote_code=config.surrogate.trust_remote_code,
        )
        surrogate_model.eval()
        for param in surrogate_model.parameters():
            param.requires_grad = False
    
    # Initialize trainer
    trainer = SurrogateTrainer(
        config=config,
        model=base_model,
        tokenizer=base_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        surrogate_model=surrogate_model,
        surrogate_tokenizer=surrogate_tokenizer,
        tpu_rank=rank,
    )
    
    try:
        trainer.train()
    finally:
        trainer.close()


def _setup_tpu_pod_environment(config: Config) -> Tuple[bool, int, int]:
    """
    Setup TPU Pod environment for multi-host training.
    
    This function detects and configures the TPU Pod environment based on
    environment variables or explicit configuration.
    
    For TPU Pods, you need to set the following environment variables on each host:
    - PJRT_DEVICE=TPU (for PJRT runtime)
    - TPU_PROCESS_ADDRESSES=host1:port,host2:port,... (comma-separated list of all hosts)
    - TPU_PROCESS_COUNT=<num_hosts> (total number of hosts)
    - TPU_PROCESS_ID=<this_host_id> (0-indexed ID of this host)
    
    Alternatively, for older XLA versions:
    - XRT_TPU_CONFIG=tpu_worker;0;host1:port|tpu_worker;1;host2:port|...
    - TPU_WORKER_ID=<this_host_id>
    
    Returns:
        Tuple of (is_tpu_pod, host_id, num_hosts)
    """
    if not HAS_XLA:
        return False, 0, 1
    
    # Check for PJRT runtime (PyTorch XLA 2.0+)
    pjrt_device = os.environ.get("PJRT_DEVICE", "").upper()
    is_pjrt = pjrt_device == "TPU"
    
    # Check for multi-host environment variables
    tpu_process_count = os.environ.get("TPU_PROCESS_COUNT", "")
    tpu_process_id = os.environ.get("TPU_PROCESS_ID", "")
    tpu_worker_id = os.environ.get("TPU_WORKER_ID", "")
    
    # Determine if this is a TPU Pod
    num_hosts = config.training.tpu_num_hosts
    host_id = 0
    
    if tpu_process_count and tpu_process_id:
        # PJRT-style environment
        num_hosts = int(tpu_process_count)
        host_id = int(tpu_process_id)
        is_tpu_pod = num_hosts > 1
        logger.info(f"Detected PJRT TPU Pod environment: host {host_id + 1}/{num_hosts}")
    elif tpu_worker_id:
        # Legacy XRT-style environment
        host_id = int(tpu_worker_id)
        is_tpu_pod = num_hosts > 1
        logger.info(f"Detected XRT TPU Pod environment: worker {host_id}/{num_hosts}")
    else:
        # Use config values
        is_tpu_pod = num_hosts > 1
        if is_tpu_pod:
            logger.warning(
                f"TPU Pod training requested ({num_hosts} hosts) but environment variables not set. "
                "Please set TPU_PROCESS_COUNT and TPU_PROCESS_ID for PJRT, "
                "or XRT_TPU_CONFIG and TPU_WORKER_ID for XRT."
            )
    
    # Initialize PJRT runtime if needed
    # Note: xr.use_spmd() is for GSPMD/tensor parallelism, not standard data parallelism.
    # For data parallel training with xmp.spawn, we do NOT enable SPMD.
    if is_pjrt and HAS_XLA_RUNTIME and config.training.tpu_use_pjrt:
        logger.info("PJRT runtime detected for TPU training")
        # PJRT is auto-initialized when PJRT_DEVICE=TPU is set
        # Validate the runtime is available
        try:
            import torch_xla.runtime as xr
            if hasattr(xr, 'device_type'):
                device_type = xr.device_type()
                logger.info(f"PJRT device type: {device_type}")
            if hasattr(xr, 'world_size'):
                pjrt_world_size = xr.world_size()
                logger.info(f"PJRT world size: {pjrt_world_size}")
        except Exception as e:
            logger.warning(f"Could not query PJRT runtime info: {e}")
    elif not is_pjrt and is_tpu_pod:
        logger.warning(
            "TPU Pod detected but PJRT runtime not enabled. "
            "For best performance, set PJRT_DEVICE=TPU environment variable."
        )

    return is_tpu_pod, host_id, num_hosts


def _train_fn_tpu_pod(
    config: Config,
    base_tokenizer,
    surrogate_tokenizer,
    train_dataset,
    eval_dataset,
    host_id: int,
    num_hosts: int,
):
    """
    Training function for TPU Pod (multi-host) training.
    
    In TPU Pod mode, each host runs this function independently.
    XLA handles the cross-host communication and gradient synchronization.
    
    Args:
        config: Training configuration
        base_tokenizer: Tokenizer for base model
        surrogate_tokenizer: Tokenizer for surrogate model (can be None)
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        host_id: This host's ID (0-indexed)
        num_hosts: Total number of hosts
    """
    # Get the XLA device (this triggers device initialization)
    device = xm.xla_device()
    
    # Get distributed info
    world_size = xm.xrt_world_size()
    global_rank = xm.get_ordinal()
    local_rank = xm.get_local_ordinal()
    
    # Set seed with rank offset
    set_seed(config.training.seed + global_rank)
    
    is_master = (global_rank == 0)
    
    if is_master:
        logger.info(f"TPU Pod training initialized:")
        logger.info(f"  - Total world size: {world_size}")
        logger.info(f"  - Number of hosts: {num_hosts}")
        logger.info(f"  - Cores per host: {world_size // num_hosts}")
    
    logger.info(f"[Host {host_id}, Rank {global_rank}] Initializing base model: {config.model.name_or_path}")
    
    # Initialize base model
    base_model = _init_model_from_config(config.model, for_training=True)
    
    surrogate_model = None
    if config.surrogate.enabled:
        logger.info(f"[Host {host_id}, Rank {global_rank}] Loading surrogate model: {config.surrogate.name_or_path}")
        surrogate_model = AutoModelForCausalLM.from_pretrained(
            config.surrogate.name_or_path,
            torch_dtype=config.surrogate.get_torch_dtype(),
            trust_remote_code=config.surrogate.trust_remote_code,
        )
        surrogate_model.eval()
        for param in surrogate_model.parameters():
            param.requires_grad = False
    
    # Initialize trainer with TPU rank info
    trainer = SurrogateTrainer(
        config=config,
        model=base_model,
        tokenizer=base_tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        surrogate_model=surrogate_model,
        surrogate_tokenizer=surrogate_tokenizer,
        tpu_rank=global_rank,
    )
    
    try:
        trainer.train()
    finally:
        trainer.close()


def _spawn_tpu_pod_training(
    config: Config,
    base_tokenizer,
    surrogate_tokenizer,
    train_dataset,
    eval_dataset,
    host_id: int,
    num_hosts: int,
):
    """
    Spawn TPU Pod training using xmp.spawn on this host.
    
    Each host spawns multiple processes (one per TPU core on this host).
    Cross-host communication is handled by XLA.
    """
    cores_per_host = config.training.tpu_cores
    
    logger.info(f"[Host {host_id}] Spawning {cores_per_host} processes for TPU cores")
    
    def _train_fn_wrapper(rank, config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset, host_id, num_hosts):
        """Wrapper that computes global rank and calls training function."""
        # Global rank = host_id * cores_per_host + local_rank
        _train_fn_tpu_pod(
            config=config,
            base_tokenizer=base_tokenizer,
            surrogate_tokenizer=surrogate_tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            host_id=host_id,
            num_hosts=num_hosts,
        )
    
    xmp.spawn(
        _train_fn_wrapper,
        args=(config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset, host_id, num_hosts),
        nprocs=cores_per_host,
        start_method='fork',
    )


def main():
    """Main entry point.
    
    Supports multiple execution modes:
    1. Single device: python train.py --config config.yaml
    2. DDP with torchrun: torchrun --nproc_per_node=4 train.py --config config.yaml
    3. TPU single-host multi-core: python train.py --config config.yaml --device tpu --tpu_cores 8
    4. TPU Pod multi-host: python train.py --config config.yaml --device tpu --tpu_cores 8 --tpu_num_hosts 4
    """
    args = parse_args()
    
    # Detect torchrun distributed training
    local_rank, global_rank, world_size = _setup_torchrun_distributed()
    is_torchrun = local_rank != -1
    
    # Load configuration
    if args.config:
        config = Config.from_yaml(args.config)
        # Override with command line arguments
        if args.base_model:
            config.model.name_or_path = args.base_model
        if hasattr(args, 'device') and args.device:
            config.training.device = args.device
        if hasattr(args, 'tpu_cores') and args.tpu_cores:
            config.training.tpu_cores = args.tpu_cores
        if hasattr(args, 'tpu_num_hosts') and args.tpu_num_hosts:
            config.training.tpu_num_hosts = args.tpu_num_hosts
        if hasattr(args, 'init_from_scratch') and args.init_from_scratch:
            config.model.init_from_scratch = True
        # Loss type overrides (command line takes precedence)
        if hasattr(args, 'standard_training') and args.standard_training:
            config.training.loss_type = "standard"
        elif hasattr(args, 'kl_divergence') and args.kl_divergence:
            config.training.loss_type = "kl"
        elif hasattr(args, 'loss_type') and args.loss_type:
            config.training.loss_type = args.loss_type
        # Handle --no_surrogate flag (forces standard loss type)
        if hasattr(args, 'no_surrogate') and args.no_surrogate:
            config.training.loss_type = "standard"
        # Ensure surrogate.enabled is consistent with loss_type
        config._sync_surrogate_enabled()
    else:
        config = Config.from_args(args)
    
    # Update local_rank from torchrun if applicable
    if is_torchrun:
        config.training.local_rank = local_rank
        logger.info(f"Using torchrun distributed training: rank {global_rank}/{world_size}")
    
    # Set random seed (with rank offset for distributed)
    seed = config.training.seed + global_rank if is_torchrun else config.training.seed
    set_seed(seed)
    
    # Create output directory (only on rank 0)
    if global_rank == 0:
        os.makedirs(config.training.output_dir, exist_ok=True)
    
    # Synchronize before continuing (for distributed)
    if is_torchrun:
        dist.barrier()
    
    # Initialize tokenizers
    if global_rank == 0:
        logger.info(f"Loading base tokenizer: {config.model.name_or_path}")
    base_tokenizer = AutoTokenizer.from_pretrained(
        config.model.name_or_path,
        trust_remote_code=config.model.trust_remote_code,
    )
    if base_tokenizer.pad_token is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token
    
    surrogate_tokenizer = None
    if config.surrogate.enabled:
        if global_rank == 0:
            logger.info(f"Loading surrogate tokenizer: {config.surrogate.name_or_path}")
        surrogate_tokenizer = AutoTokenizer.from_pretrained(
            config.surrogate.name_or_path,
            padding_side="left",
            trust_remote_code=config.surrogate.trust_remote_code,
        )
        if surrogate_tokenizer.pad_token is None:
            surrogate_tokenizer.pad_token = surrogate_tokenizer.eos_token
    
    # Load datasets
    if global_rank == 0:
        logger.info("Loading datasets...")
    train_dataset, eval_dataset = load_training_data(config.data, base_tokenizer)
    
    # Check if TPU training
    if config.training.device == "tpu":
        if not HAS_XLA:
            raise RuntimeError("PyTorch XLA required for TPU training. Install with: pip install torch-xla")
        
        # Check for TPU Pod (multi-host) training
        is_tpu_pod, host_id, num_hosts = _setup_tpu_pod_environment(config)
        
        if is_tpu_pod:
            # TPU Pod (multi-host) training
            total_cores = config.training.tpu_cores * num_hosts
            logger.info(f"Starting TPU Pod training:")
            logger.info(f"  - Hosts: {num_hosts}")
            logger.info(f"  - Cores per host: {config.training.tpu_cores}")
            logger.info(f"  - Total cores: {total_cores}")
            logger.info(f"  - This host: {host_id}")
            
            # Spawn training processes on this host
            _spawn_tpu_pod_training(
                config=config,
                base_tokenizer=base_tokenizer,
                surrogate_tokenizer=surrogate_tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                host_id=host_id,
                num_hosts=num_hosts,
            )
        elif config.training.tpu_cores > 1:
            # Single-host multi-core TPU training
            logger.info(f"Starting single-host TPU training with {config.training.tpu_cores} cores")
            
            xmp.spawn(
                _train_fn,
                args=(config, base_tokenizer, surrogate_tokenizer, train_dataset, eval_dataset),
                nprocs=config.training.tpu_cores,
                start_method='fork',
            )
        else:
            # Single TPU core training
            logger.info("Starting single TPU core training")
            
            base_model = _init_model_from_config(config.model, for_training=True)
            
            surrogate_model = None
            if config.surrogate.enabled:
                logger.info(f"Loading surrogate model: {config.surrogate.name_or_path}")
                surrogate_model = AutoModelForCausalLM.from_pretrained(
                    config.surrogate.name_or_path,
                    torch_dtype=config.surrogate.get_torch_dtype(),
                    trust_remote_code=config.surrogate.trust_remote_code,
                )
                surrogate_model.eval()
                for param in surrogate_model.parameters():
                    param.requires_grad = False
            
            trainer = SurrogateTrainer(
                config=config,
                model=base_model,
                tokenizer=base_tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                surrogate_model=surrogate_model,
                surrogate_tokenizer=surrogate_tokenizer,
            )
            
            try:
                trainer.train()
            finally:
                trainer.close()
    else:
        # Single device or torchrun DDP training (non-TPU)
        if global_rank == 0:
            if config.model.init_from_scratch:
                logger.info(f"Initializing base model from scratch: {config.model.name_or_path}")
            else:
                logger.info(f"Loading pretrained base model: {config.model.name_or_path}")
        
        base_model = _init_model_from_config(config.model, for_training=True)
        
        surrogate_model = None
        if config.surrogate.enabled:
            if global_rank == 0:
                logger.info(f"Loading surrogate model: {config.surrogate.name_or_path}")
            surrogate_model = AutoModelForCausalLM.from_pretrained(
                config.surrogate.name_or_path,
                torch_dtype=config.surrogate.get_torch_dtype(),
                trust_remote_code=config.surrogate.trust_remote_code,
            )
            surrogate_model.eval()
            for param in surrogate_model.parameters():
                param.requires_grad = False
        
        # Initialize trainer
        trainer = SurrogateTrainer(
            config=config,
            model=base_model,
            tokenizer=base_tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            surrogate_model=surrogate_model,
            surrogate_tokenizer=surrogate_tokenizer,
        )
        
        try:
            trainer.train()
        finally:
            trainer.close()


if __name__ == "__main__":
    main()