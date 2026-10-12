#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --job-name=qwen_binding
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=slurm-qwen-binding-%j.out
#SBATCH --error=slurm-qwen-binding-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load cuda12.8
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"

echo "Job: ${SLURM_JOB_ID:-local}"
echo "Host: $(hostname)"
echo "Python: $(command -v python)"
nvidia-smi -L
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA build:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see the allocated GPU")
print("GPU:", torch.cuda.get_device_name(0))
PY

MODEL_DIR="${QWEN_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Missing local model at $MODEL_DIR" >&2
  exit 1
fi

# One model process handles dev (216) and test (648), with one fresh multimodal
# message per trial. The JSONL outputs are append-only and resumable.
PYTHONUNBUFFERED=1 python -u results/run_qwen_local_experiment.py \
  --model "$MODEL_DIR" \
  --split both \
  --output-dir results/experiment_raw \
  --progress-log results/experiment_raw/progress.jsonl \
  --max-new-tokens 16

echo "Formal experiment completed. Raw outputs are in results/experiment_raw/."
python results/score_experiment.py \
  --raw-dir results/experiment_raw \
  --out-dir results/experiment_scored
echo "Scored outputs are in results/experiment_scored/."
