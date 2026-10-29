#!/usr/bin/env python3
"""Check CUDA initialization and a small GPU computation before model loading."""
import os
import sys

import torch

print(f"Python: {sys.executable}", flush=True)
print(f"Torch: {torch.__version__}; CUDA build: {torch.version.cuda}", flush=True)
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
try:
    torch.cuda.init()
    value = (torch.ones(16, device='cuda') * 2).sum().item()
    torch.cuda.synchronize()
    if value != 32:
        raise RuntimeError(f"Unexpected GPU computation result: {value}")
    print(f"CUDA check passed: {torch.cuda.get_device_name(0)}", flush=True)
except Exception as exc:
    print(f"CUDA check failed: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1)
