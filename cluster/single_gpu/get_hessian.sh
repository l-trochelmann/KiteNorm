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
python /home/ltrochelmann/LN-variants/get_hessian.py --config=$config --job_idx=$job_idx
