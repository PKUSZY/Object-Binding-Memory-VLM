#!/usr/bin/env bash
set -euo pipefail

# Install a local Qwen3-VL environment. Run this on the Linux machine that has the RTX 5090.
ENV_NAME="${QWEN_ENV_NAME:-qwen3vl}"
PYTHON_VERSION="${QWEN_PYTHON_VERSION:-3.11}"
CUDA_MODULE="${QWEN_CUDA_MODULE:-}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required. Install Miniconda/Anaconda first, then rerun this script." >&2
  exit 1
fi

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

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda activate "$ENV_NAME"
else
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
  conda activate "$ENV_NAME"
fi

python -m pip install --upgrade pip

echo "Installing CUDA-enabled PyTorch. For RTX 5090, cu128 or a newer wheel is required."
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -m pip install "transformers>=4.57.0" accelerate qwen-vl-utils pillow safetensors huggingface_hub

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("PyTorch cannot see CUDA; fix the NVIDIA driver/PyTorch installation before downloading the model")
print("GPU:", torch.cuda.get_device_name(0))
print("capability:", torch.cuda.get_device_capability(0))
PY

echo
echo "Environment ready: $ENV_NAME"
echo "Next: python results/download_qwen3vl.py --output models/Qwen3-VL-8B-Instruct"
