#!/usr/bin/env bash
set -euo pipefail

# End-to-end local path after this repository has been copied to the RTX 5090 host.
# Use --full to run all 432 calibration images; default is a 12-image pilot.
ENV_NAME="${QWEN_ENV_NAME:-qwen3vl}"
MODEL_DIR="${QWEN_MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
CUDA_MODULE="${QWEN_CUDA_MODULE:-}"
LIMIT=12
if [[ "${1:-}" == "--full" ]]; then
  LIMIT=432
fi

# On clusters, load the site CUDA module before importing torch. Example:
#   module load cuda12.8
#   QWEN_CUDA_MODULE=cuda12.8 bash results/run_qwen_pipeline.sh --full
if [[ -n "$CUDA_MODULE" ]]; then
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
      if [[ -f "$init" ]]; then
        # shellcheck disable=SC1090
        source "$init"
        break
      fi
    done
  fi
  if ! type module >/dev/null 2>&1; then
    echo "QWEN_CUDA_MODULE=$CUDA_MODULE was set, but the 'module' command is unavailable." >&2
    exit 1
  fi
  module load "$CUDA_MODULE"
fi

# A setup script cannot keep `conda activate` active in the parent shell.
# Activate the intended environment here so `bash results/run_qwen_pipeline.sh`
# does not accidentally use the system CPU-only Python.
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "Conda environment '$ENV_NAME' was not found." >&2
    echo "Run: bash results/setup_qwen3vl.sh" >&2
    exit 1
  fi
  conda activate "$ENV_NAME"
fi

echo "Python: $(command -v python)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "nvidia-smi path: $(command -v nvidia-smi || echo '<not found>')"
if command -v nvidia-smi >/dev/null 2>&1; then
  if ! nvidia-smi -L; then
    echo "nvidia-smi cannot enumerate a GPU in this job." >&2
  fi
else
  echo "nvidia-smi is not available in PATH." >&2
fi
echo "NVIDIA device files:"
ls -l /dev/nvidia* 2>/dev/null || echo "  none visible"
python - <<'PY'
import sys
print("Python version:", sys.version.split()[0])
try:
    import torch
except ImportError as exc:
    raise SystemExit(f"PyTorch is not installed in this environment: {exc}")
print("Torch:", torch.__version__, "CUDA wheel:", torch.version.cuda)
print("CUDA visible:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit(
        "CUDA is not visible. The diagnostics above distinguish an unavailable "
        "GPU node/driver from a PyTorch installation problem."
    )
print("GPU:", torch.cuda.get_device_name(0))
PY

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  echo "Model is not downloaded at $MODEL_DIR" >&2
  echo "Run: python results/download_qwen3vl.py --output $MODEL_DIR" >&2
  exit 1
fi

python results/test_qwen3vl.py --model "$MODEL_DIR"
if [[ "$LIMIT" == 12 ]]; then
  OUTPUT="results/calibration_review/qwen_local_pilot_raw.jsonl"
else
  OUTPUT="results/calibration_review/qwen_local_calibration_raw.jsonl"
fi
python results/run_qwen_local_calibration.py \
  --model "$MODEL_DIR" \
  --limit "$LIMIT" \
  --output "$OUTPUT"
if [[ "$LIMIT" == 12 ]]; then
  python results/score_calibration.py --raw "$OUTPUT" --allow-subset
else
  python results/score_calibration.py --raw "$OUTPUT" --out results/calibration_review/qwen_local_scored
fi
