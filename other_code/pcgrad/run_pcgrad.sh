#!/bin/bash
#SBATCH --account=vsharan_1861
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=pcgrad_mtl
#SBATCH --output=/scratch1/bandrede/logs/pcgrad_%j.out
#SBATCH --error=/scratch1/bandrede/logs/pcgrad_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=bandrede@usc.edu

# ─────────────────────────────────────────────────────────────────────────────
# PCGrad MTL Training Job — CSCI 567
# Submit:  sbatch pcgrad/run_pcgrad.sh   (from mtl_gradient_conflict/)
# Monitor: squeue --user=bandrede
# Log:     tail -f /scratch1/bandrede/logs/pcgrad_<JOBID>.out
# ─────────────────────────────────────────────────────────────────────────────

set -e          # fail immediately on any command error
set -o pipefail # catch errors inside pipes

echo "========================================================"
echo " PCGrad MTL — CSCI 567"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $(hostname)"
echo " Start    : $(date)"
echo "========================================================"

# ── 1. Activate venv (no module loads needed — venv is self-contained) ────────
source /home1/bandrede/envs/mtl_env/bin/activate

# Confirm correct Python and CUDA
echo "Python    : $(which python) — $(python --version)"
echo "PyTorch   : $(python -c 'import torch; print(torch.__version__, "| CUDA:", torch.cuda.is_available())')"

# ── 2. HuggingFace cache → scratch (never home1) ─────────────────────────────
export HF_HOME=/scratch1/bandrede/hf_cache
export HF_DATASETS_CACHE=/scratch1/bandrede/hf_cache/datasets
export TRANSFORMERS_CACHE=/scratch1/bandrede/hf_cache
export HF_HUB_CACHE=/scratch1/bandrede/hf_cache/hub

# ── 3. Timestamped output directory ──────────────────────────────────────────
DATE=$(date +%m%d_%H%M)
OUTPUT_DIR=/scratch1/bandrede/mtl_outputs/pcgrad_${DATE}_job${SLURM_JOB_ID}
mkdir -p "$OUTPUT_DIR"

echo "GPU       : $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Output    : $OUTPUT_DIR"
echo "--------------------------------------------------------"

# ── 4. Move into code directory ───────────────────────────────────────────────
cd /home1/bandrede/mtl_gradient_conflict/pcgrad

# ── 5. Run training ───────────────────────────────────────────────────────────
python train_pcgrad_mtl.py \
    --batch_size          32   \
    --num_epochs          20   \
    --num_workers         4    \
    --grad_signal_batches 4    \
    --steps_per_epoch     7000 \
    --eval_every          1000 \
    --patience            5    \
    --output_dir          "$OUTPUT_DIR" \
    --wandb_project       csci567-mtl   \
    --wandb_run           "pcgrad_${DATE}"

echo "--------------------------------------------------------"
echo " End      : $(date)"
echo " Results  : $OUTPUT_DIR"
echo "========================================================"
