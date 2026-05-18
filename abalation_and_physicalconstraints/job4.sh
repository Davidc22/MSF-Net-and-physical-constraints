#!/bin/bash
#SBATCH --gpus=1
#SBATCH -p gpu

module load miniforge3/24.11
module load cuda/12.4
source activate bat

python FA_integrated_final.py --region carribean --ws 21 --seeds 42 123 7 --outdir results_carribean
