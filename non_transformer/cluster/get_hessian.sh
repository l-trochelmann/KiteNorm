#!/bin/bash

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

# Job specific vars
echo "Assigning job variables"
config=$1
LR=$2

# Execute python script
echo "Executing job script"
python /home/ltrochelmann/LN-variants/non_transformer/get_init_hessian.py --config "$config" --lr $LR
