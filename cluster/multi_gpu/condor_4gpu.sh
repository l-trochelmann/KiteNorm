#!/bin/bash

export HOME=/path/to/home
export TMPDIR=/path/to/tmp

echo "Activating environment"
source /path/to/env
conda activate env-name

# Job specific vars
echo "Assigning job variables"
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
echo "Executing job script"
torchrun \
  --redirects 1:0,2:0,3:0 \
  --standalone --nnodes=1 --nproc_per_node=4 \
  /path/to/train.py --config=$config --job_idx=$job_idx