#!/usr/bin/env bash
# CPU-only work. Pass --partition=<your CPU partition> when submitting.
#SBATCH --job-name=bindmem-probes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-bindmem-probes-%j.out
#SBATCH --error=slurm-bindmem-probes-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from repository root}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1
bash scripts/run_bindmem.sh probes
