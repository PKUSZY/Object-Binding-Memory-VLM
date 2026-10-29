#!/usr/bin/env bash
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HUB_OFFLINE=1
python scripts/prepare_bindmem_priority.py
model=models/llava-1.5-7b-hf
root=runs/bindmem_priority

# Phase 1: publish a complete same-test-trial core comparison first.
python -u -m bindmem_extra.inference --backend llava --model "$model" --data data/bindmem_v2 \
  --split test --trial-ids "$root/core_test_ids.json" --output "$root/llava7b_core" \
  --reuse-from runs/bindmem_v2/llava7b_behavior
python -m bindmem_extra.score --data data/bindmem_v2 --run "$root/llava7b_core"
python -m bindmem_extra.score --data data/bindmem_v2 \
  --run-a runs/bindmem_v2/qwen8b_core runs/bindmem_v2/qwen8b_image runs/bindmem_v2/qwen8b_formats \
  --run-b "$root/llava7b_core" --allow-subset --output "$root/core_comparison"
python -m bindmem_extra.cases --data data/bindmem_v2 --runs "$root/llava7b_core" \
  --output "$root/llava7b_core/strong_error_cases"
python scripts/report_bindmem_priority.py
echo 'CORE COMPARISON READY: runs/bindmem_priority/report.md'

# Phase 2: complete matched 0–3 gradient, image and image+text queries only.
python -u -m bindmem_extra.inference --backend llava --model "$model" --data data/bindmem_gradient_n6 \
  --split test --trial-ids "$root/gradient_test_ids.json" --output "$root/llava7b_gradient"
python -m bindmem_extra.score --data data/bindmem_gradient_n6 --run "$root/llava7b_gradient"
python -m bindmem_extra.score --data data/bindmem_gradient_n6 \
  --run-a runs/bindmem_gradient_n6/qwen8b --run-b "$root/llava7b_gradient" \
  --allow-subset --output "$root/gradient_comparison"
python -m bindmem_extra.cases --data data/bindmem_gradient_n6 --runs "$root/llava7b_gradient" \
  --levels G --output "$root/llava7b_gradient/strong_error_cases"
python scripts/report_bindmem_priority.py
echo 'PRIORITY EXPERIMENT COMPLETE: runs/bindmem_priority/report.md'
