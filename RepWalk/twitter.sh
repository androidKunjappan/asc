#!/bin/sh
#SBATCH --job-name=pss # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl2_48h-1G

python3 train_cpt.py --hidden_dim 50 --dataset twitter --lr .001 --batch_size 64 --wt_decay 1e-3
