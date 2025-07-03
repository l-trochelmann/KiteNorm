#!/bin/bash

export HOME=/home/ltrochelmann
export TMPDIR=/fast/ltrochelmann/tmp

echo "Activating environment"
source /home/ltrochelmann/miniforge3/etc/profile.d/conda.sh
conda activate ln-variants

# Execute python script
echo "Executing job script"
python /home/ltrochelmann/LN-variants/CNN/main.py