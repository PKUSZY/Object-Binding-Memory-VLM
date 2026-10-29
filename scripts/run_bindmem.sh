#!/usr/bin/env bash
# Run from any directory; outputs stay under the repository by default.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

STAGE="${1:-all}"
DATA_DIR="${DATA_DIR:-data/bindmem_v2}"
RUN_DIR="${RUN_DIR:-runs/bindmem_v2/qwen3vl8b}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
GROUPS_COUNT="${GROUPS_COUNT:-320}"
# Space-separated filters. Leave empty for the complete benchmark.
read -r -a level_args <<< "${LEVELS:-}"
read -r -a question_args <<< "${QUESTIONS:-}"
filters=()
if ((${#level_args[@]})); then filters+=(--levels "${level_args[@]}"); fi
if ((${#question_args[@]})); then filters+=(--questions "${question_args[@]}"); fi
capture_args=()
if [[ "${CAPTURE:-0}" == 1 ]]; then capture_args+=(--capture --layers "${LAYERS:-auto}"); fi

prepare() {
  if [[ ! -f "$DATA_DIR/manifest.json" ]]; then
    python -m bindmem.generate --output "$DATA_DIR" --groups "$GROUPS_COUNT"
  fi
  python -m bindmem.validate --data "$DATA_DIR"
}
infer() {
  python -u -m bindmem.run --data "$DATA_DIR" --model "$MODEL_DIR" --output "$RUN_DIR" \
    --split both "${filters[@]}" "${capture_args[@]}"
}
score() {
  python -m bindmem.score --data "$DATA_DIR" --run "$RUN_DIR" --split test
  python -m bindmem.score --data "$DATA_DIR" --run "$RUN_DIR" --split dev
}
case "$STAGE" in
  prepare) prepare ;;
  run) infer ;;
  score) score ;;
  probes) python -u -m bindmem.probes --data "$DATA_DIR" --run "$RUN_DIR" ;;
  all) prepare; infer; score ;;
  *) echo "Usage: $0 {prepare|run|score|probes|all}" >&2; exit 2 ;;
esac
