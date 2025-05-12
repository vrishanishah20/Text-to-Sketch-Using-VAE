# Text-to-Sketch Generation Using Conditional VAE (CVAE)

This repository contains the implementation of a conditional variational autoencoder (CVAE) for generating sketch-style images from text prompts. The model is trained on paired text-image data and conditioned using CLIP embeddings to learn meaningful visual representations.

## Dataset
https://huggingface.co/datasets/zoheb/sketch-scene

## Architecture Overview

The model follows a standard encoder-decoder VAE structure with conditional inputs:

- **Encoder**: Convolutional layers encode the input sketch into a latent vector `z`
- **Conditioning**: CLIP-based text embeddings are concatenated with the encoder output
- **Decoder**: Reconstructs the image from the latent vector + condition
- **Loss**:
  - Binary cross-entropy for reconstruction
  - KL divergence for latent space regularization
- **Training**: Implemented PyTorch Lightning to streamline training workflows and enable faster, more scalable inference across GPUs.
- **Optimization**: Integrated mixed precision training, efficient dataloaders, and gradient clipping to accelerate convergence and stabilize training performance.

## Project Structure

```
.
├── data/                          # Contains paired sketch-text dataset files
│   └── [dataset files]            # Images and corresponding text files
│
├── inference/                     # Scripts for generating sketches from text prompts
│   └── generate.py                # Main inference script using trained model
│
├── lightning_logs/                # PyTorch Lightning logs and checkpoints
│   └── cvae/                      # Directory for CVAE model logs
│
├── logs/                          # Additional training logs and metrics
│
├── outputs/                       # Generated outputs from inference scripts
│
├── training/                      # Training scripts and utilities
│   └── train_lightning.py         # Main training script using PyTorch Lightning
│   └── model.py                   # Defines the CVAE model architecture
│   └── dataset.py                 # Dataset class for loading and preprocessing data
│   └── utils.py                   # Utility functions for training and evaluation
│
├── create_clip_embeddings.slurm   # SLURM script for generating CLIP embeddings
├── image_embeddings.npy           # Precomputed CLIP image embeddings
├── requirements.txt               # List of required Python packages
├── train.slurm                    # SLURM script for training the model
├── train_lightning.slurm          # Alternative SLURM script for training with Lightning
├── README.md                      # Project documentation (this file)
├── Report.pdf                     # Detailed project report
```

### Key Components:

- **data/**: Contains the dataset of sketches and corresponding text descriptions.
- **inference/**: Scripts for generating sketches from text prompts using the trained model.
- **lightning_logs/**: Stores logs and checkpoints from PyTorch Lightning training sessions.
- **logs/**: Additional logs and metrics from training and evaluation.
- **outputs/**: Generated sketch outputs from inference scripts.
- **training/**: Contains training scripts, model definitions, dataset classes, and utility functions.
- **create_clip_embeddings.slurm**: SLURM script to generate CLIP embeddings for the dataset.
- **image_embeddings.npy**: Precomputed CLIP image embeddings used for conditioning the model.
- **requirements.txt**: Specifies the Python package dependencies.
- **train.slurm** and **train_lightning.slurm**: SLURM scripts for submitting training jobs to a cluster.
- **README.md**: This file, providing an overview and documentation of the project.
- **Report.pdf**: A comprehensive report detailing the project's methodology and results.

## Dataset

- **Dataset**: Paired sketch-text dataset (.png, .txt)
- **Preprocessing**: Resized to 128×128, normalized, and tokenized for CLIP

## Training
## if not using colab, setup a conda env

- conda create -n tf -c conda-forge tensorflow-gpu python=3.7 &&conda activate tf

Ensure you have access to a GPU  and install dependencies:
- pip install -r requirements.txt

Train the model using:
python train_lightning.py

You can monitor training logs in `logs/` and adjust hyperparameters in the training script (e.g., batch size, latent dimension, beta).

## Inference
python reconstruction.py
python generate.py --prompt "a tree on a hill"

This loads the trained weights and uses the decoder conditioned on CLIP features of the prompt.

## SLURM (Quest HPC) Usage

If running on a SLURM-based cluster:

sbatch scripts/train_lightning.slurm


## Results (on 20 epochs)
- Reconstruction on 5 images shown
- Image generation performed with prompts
- Streamlit application deployed for visualization

## Future Work

- Training on more epochs (100-200)
- Add image embedding using CLIP so text and image are in the same state

## Acknowledgements

- OpenAI CLIP
- Reference: CVAE_CONV (https://github.com/meanna/CVAE_CONV)
- @inproceedings{fscoco,
    title={FS-COCO: Towards Understanding of Freehand Sketches of Common Objects in Context.}
    author={Chowdhury, Pinaki Nath and Sain, Aneeshan and Bhunia, Ayan Kumar and Xiang, Tao and Gryaditskaya, Yulia and Song, Yi-Zhe},
    booktitle={ECCV},
    year={2022}
}

