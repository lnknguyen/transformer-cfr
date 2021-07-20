#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --mem=2G   

module load anaconda

python3 ../src/models/mnist/run_mnist.py 