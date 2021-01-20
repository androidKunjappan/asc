#!/bin/sh
#SBATCH --job-name=all # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#python3 train.py --cpt --dataset laptop --lr .001 --batch_size 32 --phi 10.0 --entropy 2.5 --eps 0.01 --beta 0.01
#python3 train.py --cpt --dataset twitter --lr .001 --batch_size 64 --phi 10.0 --entropy 2.0 --eps 0.01 --beta 0.01 --num_epoch 40
python3 train.py --cpt --dataset twitter --lr .001 --batch_size 32 --phi 10.0 --entropy 2.5 --eps 0.01 --beta 0.01
