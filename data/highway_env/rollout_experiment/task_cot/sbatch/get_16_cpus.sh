#!/bin/bash
#SBATCH --job-name get_16_cpus        # Job name
### Logging
#SBATCH --output=%j.out                 # Stdout (%j expands to jobId)
#SBATCH --error=%j.err                  # Stderr (%j expands to jobId)
### Node info
#SBATCH --nodes=1                       # Single node or multi node
#SBATCH --time 24:00:00                 # Max time (hh:mm:ss)
#SBATCH --gres=gpu:0                    # GPUs per node
#SBATCH --mem=32G                       # Recommend 32G per GPU
#SBATCH --ntasks-per-node=1             # Tasks per node
#SBATCH --cpus-per-task=16               # Recommend 8 per GPU
### Whatever your job needs to do

export REQUESTS_CA_BUNDLE="/etc/ssl/certs"
export HTTP_PROXY="http://192.168.0.10:3128"
export HTTPS_PROXY="https://192.168.0.10:3129"

export http_proxy="http://192.168.0.10:3128"
export https_proxy="https://192.168.0.10:3129"

sleep 360000