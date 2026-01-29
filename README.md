# SDCE Training

Surrogate-Guided Cross-Entropy training for language models.

## Quick Start

```bash
# Install
pip install torch transformers datasets accelerate pyyaml wandb

# For TPU
pip install torch-xla
```

## Training Examples

### GPU - Single Device
```bash
python train.py \
  --base_model gpt2 \
  --surrogate_model Qwen/Qwen3-0.6B \
  --dataset wikitext --dataset_config wikitext-2-raw-v1 \
  --batch_size 4 --learning_rate 1e-4 --num_epochs 3
```

### GPU - Multi-GPU (DDP)
```bash
torchrun --nproc_per_node=4 train.py \
  --base_model gpt2 \
  --surrogate_model Qwen/Qwen3-0.6B \
  --dataset wikitext --dataset_config wikitext-2-raw-v1
```

### TPU - Single Host (v3-8)
```bash
python train.py \
  --device tpu --tpu_cores 8 \
  --base_model gpt2 \
  --surrogate_model Qwen/Qwen3-0.6B \
  --mixed_precision bf16 \
  --dataset wikitext --dataset_config wikitext-2-raw-v1
```

### TPU Pod (v3-32, 4 hosts)
```bash
# Set on each host:
export PJRT_DEVICE=TPU
export TPU_PROCESS_COUNT=4
export TPU_PROCESS_ID=<0-3>

python train.py \
  --device tpu --tpu_cores 8 --tpu_num_hosts 4 \
  --base_model gpt2 \
  --surrogate_model Qwen/Qwen3-0.6B \
  --mixed_precision bf16 \
  --dataset wikitext --dataset_config wikitext-2-raw-v1
```

### Standard Training (No Surrogate)
```bash
python train.py --standard_training --base_model gpt2 --dataset wikitext
```

## Config File

```yaml
# config.yaml
model:
  name_or_path: gpt2
  dtype: bfloat16
  gradient_checkpointing: true

surrogate:
  name_or_path: Qwen/Qwen3-0.6B
  dtype: bfloat16
  k: 30
  probability_threshold: 0.02
  loss_weight_initial: 1.0
  loss_weight_final: 0.0

data:
  dataset_name: wikitext
  dataset_config: wikitext-2-raw-v1
  max_seq_length: 1024

training:
  device: auto
  num_epochs: 3
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 1e-4
  mixed_precision: bf16
  loss_type: surrogate  # surrogate | kl | standard
  output_dir: ./outputs
  wandb_project: sdce-training
```

```bash
python train.py --config config.yaml
```

## Loss Types

| Type | Description |
|------|-------------|
| `surrogate` | CE + top-k perplexity-weighted surrogate loss (default) |
| `kl` | CE + KL divergence from surrogate distribution |
| `standard` | Cross-entropy only (no surrogate) |

## Key Options

| Flag | Description |
|------|-------------|
| `--surrogate_k` | Top-k tokens from surrogate (default: 30) |
| `--probability_threshold` | Min probability filter (default: 0.02) |
| `--surrogate_loss_weight_initial` | Initial surrogate weight (default: 1.0) |
| `--surrogate_loss_weight_final` | Final weight after decay (default: 0.0) |
| `--mixed_precision bf16` | Use bfloat16 (required for TPU) |
| `--gradient_checkpointing` | Reduce memory usage |
| `--init_from_scratch` | Random init instead of pretrained |

## TPU Notes

- TPU requires `bfloat16` or `float32` (no fp16 support)
- Data loading limited to 2 workers max for XLA compatibility
- Surrogate models are validated before TPU transfer

## Cloud TPU v4 Setup (GCP)

For large-scale training on Google Cloud TPU v4 pods, use the automated setup script.

### Prerequisites

1. [Google Cloud SDK](https://cloud.google.com/sdk/docs/install) installed and authenticated
2. TPU quota allocated (request via [Cloud TPU quota page](https://console.cloud.google.com/iam-admin/quotas))
3. W&B account for experiment tracking

### Quick Start (TPU v4-32)

```bash
# 1. Edit configuration in setup_tpu_training.sh
#    - Set PROJECT_ID, WANDB_API_KEY, etc.

# 2. Run full setup
./setup_tpu_training.sh setup
```

This automatically:
- Creates GCS bucket for checkpoints
- Provisions TPU v4-32 via queued-resources API
- Installs PyTorch XLA and dependencies on all workers
- Copies training code and config
- Launches distributed training

### Setup Script Commands

| Command | Description |
|---------|-------------|
| `./setup_tpu_training.sh setup` | Full setup: bucket, TPU, deps, launch training |
| `./setup_tpu_training.sh resume` | Resume setup on existing TPU (skip creation) |
| `./setup_tpu_training.sh status` | Check TPU and training status |
| `./setup_tpu_training.sh monitor` | Monitor training progress on worker 0 |
| `./setup_tpu_training.sh logs [N]` | View logs from worker N (default: 0) |
| `./setup_tpu_training.sh ssh [N]` | SSH to worker N (default: 0) |
| `./setup_tpu_training.sh stop` | Stop training on all workers |
| `./setup_tpu_training.sh cleanup` | Delete TPU (keeps bucket) |

### Configuration

Edit variables at the top of `setup_tpu_training.sh`:

```bash
export PROJECT_ID="your-gcp-project"
export TPU_NAME="tinyllama-v4-32"
export ZONE="us-central2-b"
export ACCELERATOR_TYPE="v4-32"
export WANDB_API_KEY="your-wandb-key"
```

### TPU v4-32 Training Config

The script creates `config_tinyllama_tpuv4.yaml` for TinyLlama 1.1B on SlimPajama:

| Setting | Value |
|---------|-------|
| Model | TinyLlama 1.1B |
| Surrogate | Qwen2.5-1.5B |
| Dataset | SlimPajama-627B |
| TPU | v4-32 (4 hosts × 8 chips) |
| Batch size | 4 per chip × 32 chips × 8 grad_accum = **1024** |
| Precision | bfloat16 |
| Sequence length | 2048 |

### Dependency Versions (TPU v4)

| Package | Version |
|---------|---------|
| Python | 3.10+ |
| PyTorch | 2.4.0 |
| torch_xla | 2.4.0 |
| transformers | ≥4.35.0 |
| Runtime | `tpu-ubuntu2204-base` |

### Manual TPU Pod Setup

If not using the setup script:

```bash
# 1. Create TPU via queued-resources
gcloud compute tpus queued-resources create my-tpu-qr \
  --node-id=my-tpu \
  --zone=us-central2-b \
  --accelerator-type=v4-32 \
  --runtime-version=tpu-ubuntu2204-base

# 2. Wait for ACTIVE state
gcloud compute tpus queued-resources describe my-tpu-qr \
  --zone=us-central2-b

# 3. Install deps on all workers
gcloud compute tpus tpu-vm ssh my-tpu --worker=all --command="
  pip install torch==2.4.0 torch_xla[tpu]==2.4.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html
"

# 4. Set PJRT env vars and launch on each host
export PJRT_DEVICE=TPU
export TPU_PROCESS_COUNT=4
export TPU_PROCESS_ID=<0-3>  # Different per host

python train.py --config config.yaml --device tpu --tpu_cores 8 --tpu_num_hosts 4
```

### Cleanup

```bash
# Delete TPU
./setup_tpu_training.sh cleanup

# Or manually
gcloud compute tpus queued-resources delete my-tpu-qr --zone=us-central2-b

# Delete bucket (optional)
gcloud storage rm -r gs://your-bucket-name
```
