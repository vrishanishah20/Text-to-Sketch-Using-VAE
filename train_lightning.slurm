#!/bin/bash
#SBATCH --job-name=cvae_train_300
#SBATCH --partition=gengpu
#SBATCH --gres=gpu:a100:1   
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1       
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=p32368

#SBATCH --mail-user=vrishanishah2025@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=logs/train_300_%j.out
#SBATCH --error=logs/train_300_%j.err

# === Info ===
echo "Running on node: $(hostname)"
nvidia-smi


# === Ensure logs directory exists ===
mkdir -p logs

# === Clean module environment ===
module purge
module load anaconda3

# Source Conda correctly
source /home/jci0365/miniconda3/etc/profile.d/conda.sh
conda activate vae

# === Check Python Version and Executable ===
echo "Python being used:"
which python
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import torch; print('PyTorch version:', torch.__version__)"

# === Run training using mp.spawn ===
/home/jci0365/miniconda3/envs/vae/bin/python /home/jci0365/Text-to-Sketch-Using-VAE/training/train_lightning.py

