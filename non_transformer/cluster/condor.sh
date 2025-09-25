#!/bin/bash

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

# Job specific vars
echo "Assigning job variables"
LR=$1

# Execute python script
echo "Executing with lr=${LR}"
python /home/ltrochelmann/LN-variants/non_transformer/main.py --lr $LR