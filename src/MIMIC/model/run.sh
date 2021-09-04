#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=8G   

#srun --gres=gpu:1 --time=05:30:00 --mem=8G python3 run_mimic.py trans --tune

python3 run_mimic.py lstm --tune
