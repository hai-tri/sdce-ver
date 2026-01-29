#!/bin/bash
# =============================================================================
# TPU Training Launch Script
# Run this on each TPU host with appropriate TPU_PROCESS_ID
# =============================================================================
set -e

# Get worker ID from environment or argument
WORKER_ID=${1:-${TPU_WORKER_ID:-0}}
echo "Starting training on worker ${WORKER_ID}..."

# PJRT environment for TPU v4
export PJRT_DEVICE=TPU
export TPU_PROCESS_COUNT=4
export TPU_PROCESS_ID=${WORKER_ID}

# XLA optimizations
export XLA_USE_BF16=1
export XLA_TENSOR_ALLOCATOR_MAXSIZE=100000000
export TPU_LIBRARY_PATH=/lib/libtpu.so

# Prevent memory fragmentation
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

# Disable XLA metrics for cleaner logs (enable for debugging)
export TPU_STDERR_LOG_LEVEL=0
export TF_CPP_MIN_LOG_LEVEL=2

# W&B configuration
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"
export WANDB_ENTITY="nathanngtruong-university-of-california-berkeley"
export WANDB_PROJECT="SDCE-TinyLlama"

# Only log to W&B from rank 0 to avoid duplicates
if [ "${WORKER_ID}" != "0" ]; then
    export WANDB_MODE=disabled
fi

cd ~/sdce-training

echo "Environment:"
echo "  PJRT_DEVICE=${PJRT_DEVICE}"
echo "  TPU_PROCESS_COUNT=${TPU_PROCESS_COUNT}"
echo "  TPU_PROCESS_ID=${TPU_PROCESS_ID}"
echo "  WANDB_PROJECT=${WANDB_PROJECT}"

echo "Starting training..."
python train.py \
    --config config_tinyllama_tpuv4.yaml \
    --device tpu \
    --tpu_cores 8 \
    --tpu_num_hosts 4 \
    --mixed_precision bf16

echo "Training completed on worker ${WORKER_ID}"
