#!/bin/bash
# =============================================================================
# TPU v4-32 Training Setup Script
# TinyLlama 1.1B on SlimPajama with SDCE
# =============================================================================
set -e

# =============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# =============================================================================
export PROJECT_ID="sdce-484302"
export TPU_NAME="tinyllama-v4-32"
export QR_NAME="tinyllama-qr"
export ZONE="us-central2-b"
export REGION="us-central2"
export BUCKET_NAME="sdce-tinyllama-checkpoints"
export RUNTIME_VERSION="tpu-ubuntu2204-base"
export ACCELERATOR_TYPE="v4-32"

# W&B configuration
export WANDB_API_KEY="YOUR_WANDB_API_KEY_HERE"  # Replace with your key
export WANDB_ENTITY="nathanngtruong-university-of-california-berkeley"
export WANDB_PROJECT="SDCE-TinyLlama"

# Local paths
export LOCAL_CODE_DIR="/Users/nathan/Downloads/files"
export REMOTE_CODE_DIR="~/sdce-training"

# TPU Pod configuration
export TPU_CORES=8       # Chips per host
export TPU_NUM_HOSTS=4   # Hosts for v4-32

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

check_gcloud() {
    if ! command -v gcloud &> /dev/null; then
        echo "ERROR: gcloud CLI not found. Install from https://cloud.google.com/sdk/docs/install"
        exit 1
    fi

    # Check authentication
    if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | head -n1 &> /dev/null; then
        echo "ERROR: Not authenticated. Run: gcloud auth login"
        exit 1
    fi

    log "gcloud CLI authenticated"
}

wait_for_tpu_active() {
    log "Waiting for TPU queued resource to become ACTIVE..."
    local max_attempts=60  # 30 minutes max
    local attempt=0

    while [ $attempt -lt $max_attempts ]; do
        state=$(gcloud compute tpus queued-resources describe ${QR_NAME} \
            --zone=${ZONE} \
            --project=${PROJECT_ID} \
            --format='value(state.state)' 2>/dev/null || echo "UNKNOWN")

        log "Current state: ${state} (attempt $((attempt+1))/${max_attempts})"

        case $state in
            "ACTIVE")
                log "TPU is ACTIVE and ready!"
                return 0
                ;;
            "FAILED")
                log "ERROR: TPU provisioning FAILED"
                gcloud compute tpus queued-resources describe ${QR_NAME} \
                    --zone=${ZONE} \
                    --project=${PROJECT_ID}
                exit 1
                ;;
            "SUSPENDED")
                log "ERROR: TPU request was SUSPENDED"
                exit 1
                ;;
        esac

        sleep 30
        attempt=$((attempt + 1))
    done

    log "ERROR: Timeout waiting for TPU to become active"
    exit 1
}

# =============================================================================
# STEP 1: PREREQUISITES CHECK
# =============================================================================
step1_prerequisites() {
    log "========== STEP 1: Checking Prerequisites =========="
    check_gcloud

    # Set project
    gcloud config set project ${PROJECT_ID}
    log "Project set to ${PROJECT_ID}"

    # Enable required APIs
    log "Enabling required APIs..."
    gcloud services enable compute.googleapis.com --project=${PROJECT_ID}
    gcloud services enable tpu.googleapis.com --project=${PROJECT_ID}
    gcloud services enable storage.googleapis.com --project=${PROJECT_ID}

    log "Prerequisites check complete"
}

# =============================================================================
# STEP 2: CREATE GCS BUCKET
# =============================================================================
step2_create_bucket() {
    log "========== STEP 2: Creating GCS Bucket =========="

    # Check if bucket exists
    if gcloud storage buckets describe gs://${BUCKET_NAME} --project=${PROJECT_ID} &> /dev/null; then
        log "Bucket gs://${BUCKET_NAME} already exists"
    else
        log "Creating bucket gs://${BUCKET_NAME}..."
        gcloud storage buckets create gs://${BUCKET_NAME} \
            --project=${PROJECT_ID} \
            --location=${REGION} \
            --uniform-bucket-level-access
        log "Bucket created successfully"
    fi

    # Verify access
    gcloud storage ls gs://${BUCKET_NAME} > /dev/null
    log "Bucket access verified"
}

# =============================================================================
# STEP 3: CREATE TPU VIA QUEUED RESOURCES
# =============================================================================
step3_create_tpu() {
    log "========== STEP 3: Creating TPU v4-32 =========="

    # Check if queued resource already exists
    if gcloud compute tpus queued-resources describe ${QR_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} &> /dev/null; then
        log "Queued resource ${QR_NAME} already exists"
        state=$(gcloud compute tpus queued-resources describe ${QR_NAME} \
            --zone=${ZONE} \
            --project=${PROJECT_ID} \
            --format='value(state.state)')

        if [ "$state" == "ACTIVE" ]; then
            log "TPU is already ACTIVE"
            return 0
        else
            log "Current state: ${state}"
        fi
    else
        log "Creating queued resource ${QR_NAME}..."
        gcloud compute tpus queued-resources create ${QR_NAME} \
            --node-id=${TPU_NAME} \
            --zone=${ZONE} \
            --accelerator-type=${ACCELERATOR_TYPE} \
            --runtime-version=${RUNTIME_VERSION} \
            --project=${PROJECT_ID}
        log "Queued resource created"
    fi

    # Wait for TPU to become active
    wait_for_tpu_active
}

# =============================================================================
# STEP 4: INSTALL DEPENDENCIES ON ALL TPU HOSTS
# =============================================================================
step4_install_dependencies() {
    log "========== STEP 4: Installing Dependencies =========="

    log "Installing Python packages on all ${TPU_NUM_HOSTS} workers..."
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all \
        --command='
#!/bin/bash
set -e
echo "Installing dependencies on $(hostname)..."

# Upgrade pip
pip install --upgrade pip

# Install PyTorch and PyTorch XLA for TPU v4
pip install torch==2.4.0 torch_xla[tpu]==2.4.0 \
    -f https://storage.googleapis.com/libtpu-releases/index.html

# Install training dependencies
pip install transformers>=4.35.0 \
    datasets>=2.14.0 \
    tokenizers>=0.14.0 \
    pyyaml>=6.0 \
    omegaconf>=2.3.0 \
    wandb>=0.15.0 \
    tqdm>=4.65.0 \
    numpy>=1.24.0 \
    lm-eval>=0.4.0

# Verify installations
echo "Verifying PyTorch XLA installation..."
python -c "import torch; print(f\"PyTorch version: {torch.__version__}\")"
python -c "import torch_xla; print(f\"PyTorch XLA version: {torch_xla.__version__}\")"
python -c "import torch_xla.core.xla_model as xm; print(f\"XLA device: {xm.xla_device()}\")"
python -c "import transformers; print(f\"Transformers version: {transformers.__version__}\")"

echo "Dependencies installed successfully on $(hostname)"
'

    log "Dependencies installed on all workers"
}

# =============================================================================
# STEP 5: COPY TRAINING CODE TO TPU
# =============================================================================
step5_copy_code() {
    log "========== STEP 5: Copying Training Code =========="

    # Create remote directory on all workers
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all \
        --command="mkdir -p ${REMOTE_CODE_DIR}"

    # Copy training files
    log "Copying training code to all workers..."
    gcloud compute tpus tpu-vm scp --recurse \
        ${LOCAL_CODE_DIR}/train.py \
        ${LOCAL_CODE_DIR}/losses.py \
        ${LOCAL_CODE_DIR}/requirements.txt \
        ${TPU_NAME}:${REMOTE_CODE_DIR}/ \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all

    log "Training code copied to all workers"
}

# =============================================================================
# STEP 6: CREATE CONFIGURATION FILE
# =============================================================================
step6_create_config() {
    log "========== STEP 6: Creating Configuration =========="

    # Create config file locally first
    cat > /tmp/config_tinyllama_tpuv4.yaml << 'CONFIGEOF'
# TinyLlama 1.1B on SlimPajama - TPU v4-32 Configuration
# ========================================================

model:
  name_or_path: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  dtype: "bfloat16"
  use_flash_attention: false
  gradient_checkpointing: true
  trust_remote_code: false
  init_from_scratch: false

surrogate:
  name_or_path: "Qwen/Qwen2.5-1.5B"
  dtype: "bfloat16"
  k: 30
  probability_threshold: 0.02
  enabled: true
  trust_remote_code: true
  loss_weight_initial: 1.0
  loss_weight_final: 0.0
  use_perplexity_weighting: true

data:
  dataset_name: "cerebras/SlimPajama-627B"
  dataset_config: null
  dataset_split: "train"
  eval_split: "validation"
  text_column: "text"
  max_seq_length: 2048
  preprocessing_num_workers: 2
  eval_split_ratio: 0.001
  eval_split_seed: 42

training:
CONFIGEOF

    # Append training section with variable substitution
    cat >> /tmp/config_tinyllama_tpuv4.yaml << CONFIGEOF
  output_dir: "gs://${BUCKET_NAME}/tinyllama-slimpajama"
  num_epochs: 1
  per_device_train_batch_size: 4
  per_device_eval_batch_size: 8
  gradient_accumulation_steps: 8
  learning_rate: 3.0e-4
  weight_decay: 0.1
  warmup_ratio: 0.01
  max_grad_norm: 1.0
  lr_scheduler_type: "cosine"
  loss_type: "surrogate"
  device: "tpu"
  tpu_cores: ${TPU_CORES}
  tpu_num_hosts: ${TPU_NUM_HOSTS}
  tpu_metrics_debug: false
  tpu_use_pjrt: true
  mixed_precision: "bf16"
  logging_steps: 100
  eval_steps: 5000
  save_steps: 5000
  save_total_limit: 5
  use_z_loss: true
  z_loss_multiplier: 1.0e-4
  seed: 42
  wandb_project: "${WANDB_PROJECT}"
  wandb_run_name: "tinyllama-1.1b-slimpajama-v4-32"
  wandb_entity: "${WANDB_ENTITY}"

evaluation:
  enabled: true
  eval_interval: 5000
  tasks:
    - "hellaswag"
    - "arc_easy"
    - "arc_challenge"
    - "piqa"
    - "winogrande"
  num_fewshot: 0
  batch_size: 16
  limit: 1000
  log_individual_tasks: true
  log_aggregate_score: true
CONFIGEOF

    # Copy config to all workers
    gcloud compute tpus tpu-vm scp \
        /tmp/config_tinyllama_tpuv4.yaml \
        ${TPU_NAME}:${REMOTE_CODE_DIR}/config_tinyllama_tpuv4.yaml \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all

    # Also copy locally for reference
    cp /tmp/config_tinyllama_tpuv4.yaml ${LOCAL_CODE_DIR}/config_tinyllama_tpuv4.yaml

    log "Configuration file created and copied"
}

# =============================================================================
# STEP 7: CREATE LAUNCH SCRIPT
# =============================================================================
step7_create_launch_script() {
    log "========== STEP 7: Creating Launch Script =========="

    # Create launch script with variable substitution
    cat > /tmp/launch_tpu_training.sh << LAUNCHEOF
#!/bin/bash
# =============================================================================
# TPU Training Launch Script
# Run this on each TPU host with appropriate TPU_PROCESS_ID
# =============================================================================
set -e

# Get worker ID from environment or argument
WORKER_ID=\${1:-\${TPU_WORKER_ID:-0}}
echo "Starting training on worker \${WORKER_ID}..."

# PJRT environment for TPU v4
export PJRT_DEVICE=TPU
export TPU_PROCESS_COUNT=${TPU_NUM_HOSTS}
export TPU_PROCESS_ID=\${WORKER_ID}

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
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_ENTITY="${WANDB_ENTITY}"
export WANDB_PROJECT="${WANDB_PROJECT}"

# Only log to W&B from rank 0 to avoid duplicates
if [ "\${WORKER_ID}" != "0" ]; then
    export WANDB_MODE=disabled
fi

cd ${REMOTE_CODE_DIR}

echo "Environment:"
echo "  PJRT_DEVICE=\${PJRT_DEVICE}"
echo "  TPU_PROCESS_COUNT=\${TPU_PROCESS_COUNT}"
echo "  TPU_PROCESS_ID=\${TPU_PROCESS_ID}"
echo "  WANDB_PROJECT=\${WANDB_PROJECT}"

echo "Starting training..."
python train.py \\
    --config config_tinyllama_tpuv4.yaml \\
    --device tpu \\
    --tpu_cores ${TPU_CORES} \\
    --tpu_num_hosts ${TPU_NUM_HOSTS} \\
    --mixed_precision bf16

echo "Training completed on worker \${WORKER_ID}"
LAUNCHEOF

    chmod +x /tmp/launch_tpu_training.sh

    # Copy launch script to all workers
    gcloud compute tpus tpu-vm scp \
        /tmp/launch_tpu_training.sh \
        ${TPU_NAME}:${REMOTE_CODE_DIR}/launch_tpu_training.sh \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=all

    # Also copy locally for reference
    cp /tmp/launch_tpu_training.sh ${LOCAL_CODE_DIR}/launch_tpu_training.sh

    log "Launch script created and copied"
}

# =============================================================================
# STEP 8: LAUNCH TRAINING
# =============================================================================
step8_launch_training() {
    log "========== STEP 8: Launching Training =========="

    log "Launching training on all ${TPU_NUM_HOSTS} workers..."

    # Launch on each worker with its respective ID
    for worker_id in $(seq 0 $((TPU_NUM_HOSTS - 1))); do
        log "Launching on worker ${worker_id}..."
        gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --project=${PROJECT_ID} \
            --worker=${worker_id} \
            --command="
cd ${REMOTE_CODE_DIR}
nohup bash launch_tpu_training.sh ${worker_id} > training_worker_${worker_id}.log 2>&1 &
echo 'Training launched on worker ${worker_id}, PID:' \$!
" &
    done

    # Wait for all SSH commands to complete
    wait

    log "Training launched on all workers!"
    log ""
    log "=============================================="
    log "TRAINING STARTED SUCCESSFULLY"
    log "=============================================="
    log ""
    log "Monitor with:"
    log "  ./setup_tpu_training.sh monitor"
    log ""
    log "Check W&B dashboard:"
    log "  https://wandb.ai/${WANDB_ENTITY}/${WANDB_PROJECT}"
    log ""
    log "View logs:"
    log "  ./setup_tpu_training.sh logs [worker_id]"
    log ""
}

# =============================================================================
# MONITORING COMMANDS
# =============================================================================
monitor_training() {
    log "========== Monitoring Training =========="

    # Check if training is running on worker 0
    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=0 \
        --command="
echo '=== Process Status ==='
ps aux | grep -E 'python.*train.py' | grep -v grep || echo 'No training process found'

echo ''
echo '=== Recent Logs (last 50 lines) ==='
tail -50 ${REMOTE_CODE_DIR}/training_worker_0.log 2>/dev/null || echo 'No log file found'

echo ''
echo '=== GPU/TPU Memory ==='
python -c 'import torch_xla.core.xla_model as xm; print(f\"XLA Device: {xm.xla_device()}\")' 2>/dev/null || echo 'Could not query XLA device'
"
}

view_logs() {
    local worker_id=${1:-0}
    log "Viewing logs from worker ${worker_id}..."

    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=${worker_id} \
        --command="tail -f ${REMOTE_CODE_DIR}/training_worker_${worker_id}.log"
}

check_status() {
    log "========== TPU Status =========="

    # Check queued resource status
    gcloud compute tpus queued-resources describe ${QR_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --format="table(name,state.state,tpu.nodeSpec[0].node.acceleratorType)"

    echo ""

    # Check training status on all workers
    log "Checking training process on all workers..."
    for worker_id in $(seq 0 $((TPU_NUM_HOSTS - 1))); do
        echo "Worker ${worker_id}:"
        gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --project=${PROJECT_ID} \
            --worker=${worker_id} \
            --command="ps aux | grep -E 'python.*train.py' | grep -v grep | head -1 || echo '  No training process running'" 2>/dev/null || echo "  Could not connect"
    done
}

# =============================================================================
# CLEANUP
# =============================================================================
cleanup() {
    log "========== Cleanup =========="

    read -p "Are you sure you want to delete TPU ${TPU_NAME}? (y/N) " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        log "Cleanup cancelled"
        return 0
    fi

    log "Deleting queued resource ${QR_NAME}..."
    gcloud compute tpus queued-resources delete ${QR_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --quiet || log "Queued resource may already be deleted"

    log "Cleanup complete"
    log ""
    log "Note: GCS bucket gs://${BUCKET_NAME} was NOT deleted."
    log "To delete bucket: gcloud storage rm -r gs://${BUCKET_NAME}"
}

# =============================================================================
# STOP TRAINING
# =============================================================================
stop_training() {
    log "========== Stopping Training =========="

    log "Stopping training processes on all workers..."
    for worker_id in $(seq 0 $((TPU_NUM_HOSTS - 1))); do
        log "Stopping on worker ${worker_id}..."
        gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
            --zone=${ZONE} \
            --project=${PROJECT_ID} \
            --worker=${worker_id} \
            --command="pkill -f 'python.*train.py' || echo 'No process to kill'" 2>/dev/null &
    done
    wait

    log "Training stopped on all workers"
}

# =============================================================================
# SSH TO WORKER
# =============================================================================
ssh_worker() {
    local worker_id=${1:-0}
    log "SSH to worker ${worker_id}..."

    gcloud compute tpus tpu-vm ssh ${TPU_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --worker=${worker_id}
}

# =============================================================================
# FULL SETUP (ALL STEPS)
# =============================================================================
full_setup() {
    log "========== FULL TPU TRAINING SETUP =========="
    log "TPU Name: ${TPU_NAME}"
    log "Accelerator: ${ACCELERATOR_TYPE}"
    log "Zone: ${ZONE}"
    log "Project: ${PROJECT_ID}"
    log "Bucket: gs://${BUCKET_NAME}"
    log "=============================================="
    echo ""

    step1_prerequisites
    echo ""

    step2_create_bucket
    echo ""

    step3_create_tpu
    echo ""

    step4_install_dependencies
    echo ""

    step5_copy_code
    echo ""

    step6_create_config
    echo ""

    step7_create_launch_script
    echo ""

    step8_launch_training
}

# =============================================================================
# RESUME SETUP (SKIP TPU CREATION)
# =============================================================================
resume_setup() {
    log "========== RESUME SETUP (TPU exists) =========="

    # Verify TPU is active
    state=$(gcloud compute tpus queued-resources describe ${QR_NAME} \
        --zone=${ZONE} \
        --project=${PROJECT_ID} \
        --format='value(state.state)' 2>/dev/null || echo "NOT_FOUND")

    if [ "$state" != "ACTIVE" ]; then
        log "ERROR: TPU is not active (state: ${state})"
        log "Run './setup_tpu_training.sh setup' for full setup"
        exit 1
    fi

    step4_install_dependencies
    echo ""

    step5_copy_code
    echo ""

    step6_create_config
    echo ""

    step7_create_launch_script
    echo ""

    step8_launch_training
}

# =============================================================================
# USAGE
# =============================================================================
usage() {
    cat << EOF
Usage: $0 <command> [options]

Commands:
  setup           Full setup: create bucket, TPU, install deps, launch training
  resume          Resume setup on existing TPU (skip TPU creation)

  create-bucket   Create GCS bucket only
  create-tpu      Create TPU only
  install-deps    Install dependencies on TPU
  copy-code       Copy training code to TPU
  create-config   Create configuration file
  launch          Launch training on all workers

  status          Check TPU and training status
  monitor         Monitor training progress
  logs [worker]   View logs from worker (default: 0)
  ssh [worker]    SSH to worker (default: 0)

  stop            Stop training on all workers
  cleanup         Delete TPU (keeps bucket)

Examples:
  $0 setup                    # Full setup and launch
  $0 status                   # Check status
  $0 logs 0                   # View logs from worker 0
  $0 ssh 2                    # SSH to worker 2
  $0 stop                     # Stop training
  $0 cleanup                  # Delete TPU

EOF
}

# =============================================================================
# MAIN
# =============================================================================
case "${1:-}" in
    setup)
        full_setup
        ;;
    resume)
        resume_setup
        ;;
    create-bucket)
        step2_create_bucket
        ;;
    create-tpu)
        step1_prerequisites
        step3_create_tpu
        ;;
    install-deps)
        step4_install_dependencies
        ;;
    copy-code)
        step5_copy_code
        ;;
    create-config)
        step6_create_config
        ;;
    launch)
        step8_launch_training
        ;;
    status)
        check_status
        ;;
    monitor)
        monitor_training
        ;;
    logs)
        view_logs "${2:-0}"
        ;;
    ssh)
        ssh_worker "${2:-0}"
        ;;
    stop)
        stop_training
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        usage
        ;;
    *)
        if [ -n "${1:-}" ]; then
            echo "Unknown command: $1"
            echo ""
        fi
        usage
        exit 1
        ;;
esac
