#!/bin/bash
#SBATCH --job-name obs_reconst        # Job name
### Logging
#SBATCH --output=%j.out                 # Stdout (%j expands to jobId)
#SBATCH --error=%j.err                  # Stderr (%j expands to jobId)
### Node info
#SBATCH --nodes=1                       # Single node or multi node
#SBATCH --time 24:00:00                 # Max time (hh:mm:ss)
#SBATCH --gres=gpu:4                    # GPUs per node
#SBATCH --mem=128G                       # Recommend 32G per GPU
#SBATCH --ntasks-per-node=4             # Tasks per node
#SBATCH --cpus-per-task=8               # Recommend 8 per GPU
### Whatever your job needs to do

conda activate vla_hw
cd /u/shuhan/projects/vla/scripts/fast_train
python train_cont_obs_token_action_vla.py --exp_name action_only --num_epochs 50 --batch_size 128 --action_weight 1.0 --obs_weight 0.0 --reconst_weight 0.0