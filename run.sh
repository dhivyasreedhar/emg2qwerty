#!/bin/bash
#SBATCH -p GPU                  # Use the GPU partition
#SBATCH --gres=gpu:v100-32:8     # Request 8 V100 GPUs
#SBATCH -t 48:00:00              # Set a 48-hour time limit
#SBATCH -A cis250053p            # Charge job to your group account
#SBATCH --job-name=train_emg     # Set a job name
#SBATCH --output=logs/%x-%j.out  # Save output logs in logs/ folder
#SBATCH --error=logs/%x-%j.err   # Save error logs separately

# Load necessary modules
#module load cuda/12.6.1          # Ensure correct CUDA version
#module load python/3.8.6         # Use the correct Python version
#module load anaconda3


#conda env create -f environment.yml
#conda activate emg2qwerty
#pip install hydra-core --quiet
#pip install -e .

#ln -s ../../shared/emg2qwerty-data-2021-08 ~/emg2qwerty/data

#python scripts/print_dataset_stats.py
source ~/miniconda3/etc/profile.d/conda.sh
conda activate emg2qwerty

#python -m emg2qwerty.train \
 # +user=generic \
  #+trainer.accelerator=gpu +trainer.devices=8

python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu trainer.devices=8 \
  --multirun

