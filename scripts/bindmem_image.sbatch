#!/usr/bin/env bash
# Submit from the repository root. Re-submission resumes completed trials.
# Temporarily avoid nodes implicated in the previous failed submissions.
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --exclude=node001,node015
#SBATCH --job-name=bindmem-image
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=slurm-bindmem-image-%j.out
#SBATCH --error=slurm-bindmem-image-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit with sbatch from the repository root}"
if [[ ! -f bindmem/run.py ]]; then
  echo "Submit from /ceph/home/shizhongyu/class_project" >&2
  exit 1
fi

if ! type module >/dev/null 2>&1; then
  for init in /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
    if [[ -f "$init" ]]; then source "$init"; break; fi
  done
fi
module load "${QWEN_CUDA_MODULE:-cuda12.8}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${QWEN_ENV_NAME:-qwen3vl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-10}"
export PYTHONUNBUFFERED=1

echo "Job: ${SLURM_JOB_ID:-local}; node: $(hostname); started: $(date -Is)"
nvidia-smi -L
if python scripts/check_bindmem_cuda.py; then
  echo "Using the current CUDA library environment."
elif env -u LD_LIBRARY_PATH python scripts/check_bindmem_cuda.py; then
  # Apply this fallback only after a fresh process verifies GPU computation.
  unset LD_LIBRARY_PATH
  echo "CUDA check passed without LD_LIBRARY_PATH; using that environment for this job."
else
  echo "CUDA failed in both environments. No inference was started; inspect this job's logs." >&2
  exit 1
fi

DATA_DIR="${DATA_DIR:-data/bindmem_v2}"
MODEL_DIR="${MODEL_DIR:-models/Qwen3-VL-8B-Instruct}"
RUN_DIR="${RUN_DIR:-runs/bindmem_v2/qwen8b_image}"

# Keep the existing image-run settings so its manifest and completed trials match.
python -u -m bindmem.run \
  --data "$DATA_DIR" \
  --model "$MODEL_DIR" \
  --output "$RUN_DIR" \
  --split both \
  --levels L2 L3 \
  --questions Q1

python -m bindmem.score --data "$DATA_DIR" --run "$RUN_DIR" --split test
python -m bindmem.score --data "$DATA_DIR" --run "$RUN_DIR" --split dev
echo "Image inference and scoring completed: $RUN_DIR/scored; finished: $(date -Is)"
