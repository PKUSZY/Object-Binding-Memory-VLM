#!/usr/bin/env bash
#SBATCH --partition=IAI_SLURM_3090
#SBATCH --exclude=node001,node014,node015
#SBATCH --job-name=bindmem-extra
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --qos=singlegpu
#SBATCH --cpus-per-task=10
#SBATCH --time=48:00:00
#SBATCH --output=slurm-bindmem-extra-%j.out
#SBATCH --error=slurm-bindmem-extra-%j.err
set -Eeuo pipefail
trap 'rc=$?; echo "Failed: status=$rc line=$LINENO node=$(hostname) time=$(date -Is)" >&2; exit "$rc"' ERR
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
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
export HF_HOME="${HF_HOME:-$PWD/.cache/huggingface}"
nvidia-smi -L
if python scripts/check_bindmem_cuda.py; then
  echo 'CUDA check passed.'
elif env -u LD_LIBRARY_PATH python scripts/check_bindmem_cuda.py; then
  unset LD_LIBRARY_PATH
else
  echo 'CUDA failed; no inference started.' >&2; exit 1
fi
bash scripts/run_bindmem_extra.sh
