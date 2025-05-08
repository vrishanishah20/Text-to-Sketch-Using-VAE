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

## License

MIT License. Feel free to use and adapt for academic or personal use.
