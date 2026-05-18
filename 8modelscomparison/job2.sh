#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu

module load miniforge3/24.11
module load cuda/12.4
source activate bat

python MSFNet_modelcomparison_carribean.py
