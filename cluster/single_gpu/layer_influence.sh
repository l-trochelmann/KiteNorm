#!/bin/bash

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

# Job specific vars
echo "Assigning job variables"
ckpt=$1

# Execute python script
echo "Executing job script"
python /home/ltrochelmann/LN-variants/layer_influence.py  --checkpoint_path=$ckpt --wandb_project="LN-variants" --wandb_dir="/home/ltrochelmann/LN-variants/logs/wandb"
