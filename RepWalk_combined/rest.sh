#!/bin/sh
#SBATCH --job-name=pss # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl2_48h-1G

python3 train.py --hidden_dim 50 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-2
python3 train.py --hidden_dim 50 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-3
python3 train.py --hidden_dim 50 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-4
python3 train.py --hidden_dim 50 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-5
python3 train.py --hidden_dim 50 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-6

python3 train.py --hidden_dim 100 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-2
python3 train.py --hidden_dim 100 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-3
python3 train.py --hidden_dim 100 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-4
python3 train.py --hidden_dim 100 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-5
python3 train.py --hidden_dim 100 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-6

python3 train.py --hidden_dim 300 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-2
python3 train.py --hidden_dim 300 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-3
python3 train.py --hidden_dim 300 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-4
python3 train.py --hidden_dim 300 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-5
python3 train.py --hidden_dim 300 --cpt --dataset restaurant --lr .001 --batch_size 25 --wt_decay 1e-6