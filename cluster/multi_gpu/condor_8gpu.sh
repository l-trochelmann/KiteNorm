#!/bin/bash

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

# Job specific vars
echo "Assigning job variables"
config=$1
job_idx=$2 # CONDOR job arrays range from 0 to n-1

# Execute python script
echo "Executing job script"
torchrun \
  --redirects 1:0,2:0,3:0,4:0,5:0,6:0,7:0 \
  --standalone --nnodes=1 --nproc_per_node=8 \
  /home/ltrochelmann/LN-variants/train.py --config=$config --job_idx=$job_idx