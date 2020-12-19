#!/bin/sh
#SBATCH --job-name=pss # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --hidden_dim 50 --lr .001 --batch_size 64 --dataset_name 14semeval_laptop
python3 main.py --hidden_dim 50 --lr .0005 --batch_size 64 --dataset_name 14semeval_laptop
python3 main.py --hidden_dim 50 --lr .003 --batch_size 64 --dataset_name 14semeval_laptop
python3 main.py --hidden_dim 100 --lr .001 --batch_size 64 --dataset_name 14semeval_laptop
python3 main.py --hidden_dim 300 --lr .001 --batch_size 64 --dataset_name 14semeval_laptop
python3 main.py --hidden_dim 50 --lr .001 --batch_size 25 --dataset_name 14semeval_laptop

