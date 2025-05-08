import tensorflow as tf
import torch
from ConvolutionalCondVAE import ConvCVAE, Decoder, Encoder
from data_preprocessing import SketchDataset
from train import train

"""The encoder will take both the image and the text. The image is processed through a CNN and CLIP, 
and the text is processed through a Transformer-based architecture (such as BERT) to extract features. 
These features are then concatenated and passed to the latent space."""

# Setting device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = SketchDataset()

# Hyperparameters
latent_dim = 128
label_dim = 50  # Max token length after padding
image_dim = [128, 128, 3]  # Image size (128x128, RGB)
beta = 0.65  # Beta-VAE hyperparameter

#Initilizing encoder and decoder
encoder = Encoder(latent_dim=128, label_dim=768)
decoder = Decoder()

model = ConvCVAE(
    encoder,
    decoder,
    label_dim=label_dim, #conditioning on labels
    latent_dim=latent_dim,
    image_dim=image_dim,
    beta=beta
)

# Training hyperparameters
epochs = 20
batch_size = 32
learning_rate = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

model.to(device)

train(model, dataset, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
