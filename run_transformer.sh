#!/bin/bash
#SBATCH -p GPU                  # Use the GPU partition
#SBATCH --gres=gpu:v100-32:8     # Request 8 V100 GPUs
#SBATCH -t 48:00:00              # Set a 48-hour time limit
#SBATCH -A cis250053p            # Charge job to your group account
#SBATCH --job-name=train_emg     # Set a job name
#SBATCH --output=logs/%x-%j.out  # Save output logs in logs/ folder
#SBATCH --error=logs/%x-%j.err   # Save error logs separately

# Load necessary modules
module load cuda/12.6.1          # Keep CUDA if needed

# Activate Conda
source ~/miniconda3/bin/activate
conda activate emg2qwerty

# Optional: double-check Python & hydra
echo "Using Python from: $(which python)"
echo "Using pip from: $(which pip)"
echo "Python version: $(python --version)"
python -c "import hydra; print('Hydra imported successfully:', hydra.__version__)"

# Resume training
python -m emg2qwerty.train \
  user=generic \
  trainer.accelerator=gpu \
  trainer.devices=8 \
  cluster=slurm \
  +trainer.enable_progress_bar=true
