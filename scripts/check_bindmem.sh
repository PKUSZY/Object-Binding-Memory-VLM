#!/usr/bin/env bash
# Small CPU protocol checks; no large model or GPU is used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python -m compileall -q bindmem tests
python -m unittest discover -s tests -v
PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}" python tests/check_qwen_capture.py
PYTHONPATH="${PWD}${PYTHONPATH:+:$PYTHONPATH}" python tests/check_probe_pipeline.py
