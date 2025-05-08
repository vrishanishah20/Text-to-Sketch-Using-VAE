#!/bin/bash
#SBATCH --job-name=clip_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=1
#SBATCH --mem=400G
#SBATCH --time=20:00:00
#SBATCH --partition=gengpu
#SBATCH --account=p32368

#SBATCH --mail-user=vrishanishah2025@u.northwestern.edu
#SBATCH --mail-type=BEGIN,END,FAIL

#SBATCH --output=logs/clip_embed_%j.out
#SBATCH --error=logs/clip_embed_%j.err

# === Info ===
echo "Running on node: $(hostname)"
nvidia-smi


# === Ensure logs directory exists ===
mkdir -p logs

# Initialize Conda and activate the environment 
# Ensure the conda environment is initialized properly in the SLURM job
module purge
module load anaconda3
source /home/jci0365/miniconda3/etc/profile.d/conda.sh
conda activate vae
# Check if Conda environment is activated
if [ $? -ne 0 ]; then
  echo "Error: Failed to activate conda environment 'vae'."
  exit 1
fi

echo "Successfully activated Conda environment 'vae'."

# Check Python Version and Executable 
echo "Python being used:"
which python
python -c "import sys; print('Python executable:', sys.executable)"
python -c "import torch; print('PyTorch version:', torch.__version__)"

# Run training
/home/jci0365/miniconda3/envs/vae/bin/python /home/jci0365/Text-to-Sketch-Using-VAE/training/create_clip_embeddings.py

