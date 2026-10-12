#!/usr/bin/env bash
set -u

if [[ -n "${QWEN_CUDA_MODULE:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
      if [[ -f "$init" ]]; then source "$init"; break; fi
    done
  fi
  if type module >/dev/null 2>&1; then
    echo "Loading CUDA module: $QWEN_CUDA_MODULE"
    module load "$QWEN_CUDA_MODULE"
  else
    echo "Could not initialize the environment-modules command" >&2
  fi
fi

echo "== executable paths =="
command -v nvidia-smi || true
command -v conda || true
command -v python || true
python - <<'PY'
import sys
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("torch CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("capability:", torch.cuda.get_device_capability(0))
except Exception as exc:
    print("torch check failed:", type(exc).__name__, exc)
PY

echo "== nvidia-smi =="
if ! nvidia-smi; then
  echo
  echo "nvidia-smi failed. This is a driver or GPU passthrough problem, not a Qwen model problem."
  echo "Run this script on the host/container where the RTX 5090 is installed and exposed."
  exit 1
fi
