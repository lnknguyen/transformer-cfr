#!/bin/bash
# SBATCH --gres=gpu:1
# SBATCH --time=01:30:00
# SBATCH --mem=8G   

module load anaconda
python3 run_mnist.py 
