#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --time=08:30:00
#SBATCH --mem=8G   

# srun --gres=gpu:v100:1 --time=01:30:00 --mem=8G python3 run_mnist.py

#module load anaconda
python3 run_mnist.py lstm --tune
#python3 run_mnist.py trans --tune
