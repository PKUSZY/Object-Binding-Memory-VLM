#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen_text_probe
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=slurm-qwen-text-probe-%j.out
#SBATCH --error=slurm-qwen-text-probe-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"
nvidia-smi -L
MODEL_DIR="${QWEN_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
python results/generate_text_query_benchmark.py
PYTHONUNBUFFERED=1 python -u results/run_qwen_text_query_experiment.py --model "$MODEL_DIR"
python results/score_text_query_experiment.py
