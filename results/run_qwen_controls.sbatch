#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen_controls
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=slurm-qwen-controls-%j.out
#SBATCH --error=slurm-qwen-controls-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"
MODEL_DIR="${QWEN_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
python results/generate_controls.py
PYTHONUNBUFFERED=1 python -u results/run_qwen_control_experiment.py --model "$MODEL_DIR"
