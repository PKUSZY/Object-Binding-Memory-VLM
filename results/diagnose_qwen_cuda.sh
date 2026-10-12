#!/usr/bin/env bash
set -u

echo "== environment =="
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "CUDA_HOME=${CUDA_HOME:-<unset>}"
echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-<unset>}"
echo
echo "== driver =="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>&1 || true
echo
echo "== python =="
command -v python || true
python - <<'PY'
import os, sys, traceback
print("python:", sys.executable)
try:
    import torch
    print("torch:", torch.__version__)
    print("torch CUDA build:", torch.version.cuda)
    print("torch compiled CUDA:", torch.backends.cuda.is_built())
    print("device count before init:", torch.cuda.device_count())
    try:
        torch.cuda.init()
        print("init: OK")
        print("device count after init:", torch.cuda.device_count())
        print("device:", torch.cuda.get_device_name(0))
    except Exception:
        print("init: FAILED")
        traceback.print_exc()
except Exception:
    traceback.print_exc()
PY
echo
echo "== retry without module library path =="
env -u LD_LIBRARY_PATH python - <<'PY'
import torch
print("torch:", torch.__version__, "build:", torch.version.cuda)
print("CUDA available without LD_LIBRARY_PATH:", torch.cuda.is_available())
if torch.cuda.is_available(): print("GPU:", torch.cuda.get_device_name(0))
PY
