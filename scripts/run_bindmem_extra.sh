#!/usr/bin/env bash
# Launch from the shared repository. All outputs are separate from completed v2 runs.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
MODEL_KIND="${MODEL_KIND:-llava}"
SUITE="${SUITE:-all}"
STAGE="${STAGE:-all}"
GROUPS_COUNT="${GROUPS_COUNT:-320}"
DATA_DIR="${DATA_DIR:-data/bindmem_v2}"
read -r -a gradient_sizes <<< "${GRADIENT_OBJECTS:-6}"
case "$MODEL_KIND" in
  llava) MODEL_DIR="${MODEL_DIR:-models/llava-1.5-7b-hf}"; MODEL_TAG=llava7b ;;
  qwen) MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"; MODEL_TAG=qwen8b ;;
  *) echo 'MODEL_KIND must be llava or qwen' >&2; exit 2 ;;
esac
[[ "$SUITE" == all || "$SUITE" == gradient || "$SUITE" == v2 ]] || { echo 'SUITE must be all, gradient or v2'; exit 2; }
[[ "$STAGE" == all || "$STAGE" == prepare || "$STAGE" == run || "$STAGE" == score ]] || { echo 'STAGE must be all, prepare, run or score'; exit 2; }
for n in "${gradient_sizes[@]}"; do [[ "$n" == 4 || "$n" == 6 ]] || { echo 'GRADIENT_OBJECTS must contain 4 and/or 6'; exit 2; }; done

prepare_gradient() {
  local n="$1" gradient_dir="data/bindmem_gradient_n$1"
  mkdir -p data
  # Separate models may start concurrently; only one process may generate a shared dataset.
  (
    flock -x 9
    if [[ ! -f "$gradient_dir/manifest.json" ]]; then
      python -m bindmem_extra.gradient --output "$gradient_dir" --groups "$GROUPS_COUNT" --objects "$n"
    fi
    python -m bindmem_extra.gradient --output "$gradient_dir" --validate-only
    python - "$gradient_dir" "$GROUPS_COUNT" "$n" <<'PY'
import json,sys
from pathlib import Path
m=json.loads((Path(sys.argv[1])/'manifest.json').read_text())
assert m['groups']==int(sys.argv[2]) and m['total_objects']==int(sys.argv[3]) and m['seed']==20261028, 'Existing gradient settings differ'
PY
  ) 9>"data/.gradient_n${n}.lock"
}
score_run() {
  local data="$1" run="$2"
  python -m bindmem_extra.score --data "$data" --run "$run" --split test
  python -m bindmem_extra.score --data "$data" --run "$run" --split dev
}
run_dataset() {
  local data="$1" run="$2"
  python -m bindmem_extra.inference --backend "$MODEL_KIND" --model "$MODEL_DIR" --data "$data" \
    --output "${run}_preflight" --context-policy exclude --preflight
  python -u -m bindmem_extra.inference --backend "$MODEL_KIND" --model "$MODEL_DIR" --data "$data" \
    --output "${run}_smoke" --context-policy exclude --smoke --split dev
  python -m bindmem_extra.score --data "$data" --run "${run}_smoke" --split dev
  python -u -m bindmem_extra.inference --backend "$MODEL_KIND" --model "$MODEL_DIR" --data "$data" \
    --output "$run" --context-policy exclude
}
if [[ "$STAGE" == all || "$STAGE" == prepare ]]; then
  if [[ "$MODEL_KIND" == llava ]]; then python scripts/prepare_llava.py --output "$MODEL_DIR"; fi
  if [[ "$SUITE" == all || "$SUITE" == v2 ]]; then python -m bindmem.validate --data "$DATA_DIR"; fi
  if [[ "$SUITE" == all || "$SUITE" == gradient ]]; then
    for n in "${gradient_sizes[@]}"; do prepare_gradient "$n"; done
  fi
fi
[[ "$STAGE" != prepare ]] || exit 0
if [[ "$SUITE" == all || "$SUITE" == v2 ]]; then
  run="runs/bindmem_v2/${MODEL_TAG}_behavior"
  if [[ "$STAGE" != score ]]; then run_dataset "$DATA_DIR" "$run"; fi
  score_run "$DATA_DIR" "$run"
  python -m bindmem_extra.cases --data "$DATA_DIR" --runs "$run" --output "$run/strong_error_cases"
  if [[ "$MODEL_KIND" == llava ]]; then
    qwen_runs=(runs/bindmem_v2/qwen8b_core runs/bindmem_v2/qwen8b_image runs/bindmem_v2/qwen8b_choices runs/bindmem_v2/qwen8b_formats runs/bindmem_v2/qwen8b_basic runs/bindmem_v2/qwen8b_extensions)
    python -m bindmem_extra.score --data "$DATA_DIR" --run-a "${qwen_runs[@]}" --run-b "$run" \
      --allow-subset --output runs/bindmem_v2/qwen8b_vs_llava7b
  fi
fi
if [[ "$SUITE" == all || "$SUITE" == gradient ]]; then
  for n in "${gradient_sizes[@]}"; do
    data="data/bindmem_gradient_n$n"; run="runs/bindmem_gradient_n$n/${MODEL_TAG}"
    if [[ "$STAGE" != score ]]; then run_dataset "$data" "$run"; fi
    score_run "$data" "$run"
    python -m bindmem_extra.cases --data "$data" --runs "$run" --levels G --output "$run/strong_error_cases"
    if [[ -f "runs/bindmem_gradient_n$n/qwen8b/scored/test/summary.json" && -f "runs/bindmem_gradient_n$n/llava7b/scored/test/summary.json" ]]; then
      python -m bindmem_extra.score --data "$data" \
        --run-a "runs/bindmem_gradient_n$n/qwen8b" --run-b "runs/bindmem_gradient_n$n/llava7b" \
        --output "runs/bindmem_gradient_n$n/qwen8b_vs_llava7b"
    fi
  done
fi
echo "Finished: model=$MODEL_KIND suite=$SUITE stage=$STAGE"
