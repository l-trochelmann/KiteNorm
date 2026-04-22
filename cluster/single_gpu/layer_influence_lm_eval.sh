#!/bin/bash

set -euo pipefail

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

echo "Assigning job variables"
ckpt=${1:?missing checkpoint path}
limit=${2:-none}
tasks=${3:-piqa,arc_easy,winogrande}
batch_size=${4:-8}
device=${5:-auto}
hf_cache_dir=${6:-/fast/ltrochelmann/data/lm/lm-eval}

job_scratch=${_CONDOR_SCRATCH_DIR:-/tmp}
local_hf_cache="$job_scratch/lm-eval-cache"
mkdir -p "$local_hf_cache"

echo "Copying lm-eval cache from $hf_cache_dir to $local_hf_cache"
rsync -a "$hf_cache_dir/" "$local_hf_cache/"

export HF_HOME=$local_hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HF_HUB_CACHE=$HF_HOME/hub
export HF_MODULES_CACHE=$HF_HOME/modules
export TRANSFORMERS_CACHE=$HF_HOME/transformers
mkdir -p "$HF_DATASETS_CACHE" "$HF_HUB_CACHE" "$HF_MODULES_CACHE" "$TRANSFORMERS_CACHE"

echo "HF_HOME=$HF_HOME"
echo "HF_DATASETS_CACHE=$HF_DATASETS_CACHE"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "HF_MODULES_CACHE=$HF_MODULES_CACHE"

extra_args=()
if [[ "$limit" != "none" && "$limit" != "null" && -n "$limit" ]]; then
  extra_args+=(--limit "$limit")
fi

echo "Executing job script"
python "$REPO_ROOT/layer_influence_lm-eval.py" \
  --checkpoint_path="$ckpt" \
  --wandb_project="LN-variants" \
  --wandb_dir="$REPO_ROOT/logs/wandb" \
  --tasks="$tasks" \
  --batch_size="$batch_size" \
  --device="$device" \
  --hf_cache_dir="$local_hf_cache" \
  "${extra_args[@]}"
