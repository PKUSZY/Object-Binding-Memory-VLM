#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen_calib
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=12:00:00
#SBATCH --output=slurm-qwen-calib-%j.out
#SBATCH --error=slurm-qwen-calib-%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"
nvidia-smi -L
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, "available:", torch.cuda.is_available())
if not torch.cuda.is_available(): raise SystemExit("CUDA unavailable")
PY
MODEL_DIR="${QWEN_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
PYTHONUNBUFFERED=1 python -u results/run_qwen_local_calibration.py --model "$MODEL_DIR" --limit 432 --output results/calibration_review/qwen_local_calibration_raw.jsonl
python results/score_calibration.py --raw results/calibration_review/qwen_local_calibration_raw.jsonl --out results/calibration_review/qwen_local_scored
