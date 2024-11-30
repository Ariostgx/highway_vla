#!/bin/bash
#SBATCH --job-name get_3_gpus        # Job name
### Logging
#SBATCH --output=%j.out                 # Stdout (%j expands to jobId)
#SBATCH --error=%j.err                  # Stderr (%j expands to jobId)
### Node info
#SBATCH --nodes=1                       # Single node or multi node
#SBATCH --time 48:00:00                 # Max time (hh:mm:ss)
#SBATCH --gres=gpu:3                    # GPUs per node
#SBATCH --mem=96G                       # Recommend 32G per GPU
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=32               # Recommend 8 per GPU
### Whatever your job needs to do

sleep 360000