"""
Creating the conditional Conv VAE, the labels are added while creating the encoder layers or added later based on the use_cond_input parameter.
The VAE learns the conditonal latent representation of the images and its linked to the labels. 
The mean and variance (z_mean and z_log_var), denote the latent repn Z and its given to the decoder to regenerate the image
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


#ENCODER

class Encoder(nn.Module):
    """ReLU activation functions have a tendency to cause vanishing gradients (if the input values are too small) 
        or exploding gradients (if the input values are too large). The he_normal() initializer is specifically designed
        to handle these issues by ensuring that the variance of the weights is appropriately scaled for layers with ReLU activations."""
    def __init__(self, latent_dim, label_dim=768, concat_input_and_condition=True):
        super(Encoder, self).__init__()
        self.use_cond_input = concat_input_and_condition
        self.label_dim = label_dim 

        self.enc_block_1 = nn.Conv2d(3 + self.label_dim, 32, kernel_size=3, stride=2, padding=1)
        self.enc_block_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.enc_block_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.enc_block_4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.enc_block_5 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)
        self.enc_block_6 = nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.dense = nn.Linear(1024 * 2 * 2, latent_dim * 2) #2 outputs mean and log variance

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn6 = nn.BatchNorm2d(1024)

    def forward(self, input_image, input_label, conditional_input, latent_dim):
        
        x = input_image
        # Expand labels and concatenate along channel axis
        labels = input_label.view(-1, input_label.size(1), 1, 1)  # [B, label_dim, 1, 1]
        labels = labels.expand(-1, -1, x.size(2), x.size(3))      # [B, label_dim, H, W]
        x = torch.cat([x, labels], dim=1)  # [B, 3+label_dim, H, W] 

        x = F.leaky_relu(self.bn1(self.enc_block_1(x)))
        x = F.leaky_relu(self.bn2(self.enc_block_2(x)))
        x = F.leaky_relu(self.bn3(self.enc_block_3(x)))
        x = F.leaky_relu(self.bn4(self.enc_block_4(x)))

        if not self.use_cond_input:
            labels = input_label.view(-1, input_label.size(1), 1, 1)
            labels = labels.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, labels], dim=1)

        x = F.leaky_relu(self.bn5(self.enc_block_5(x)))
        x = F.leaky_relu(self.bn6(self.enc_block_6(x)))
        x = self.flatten(x)
        x = self.dense(x)
        return x

    
    
#DECODER

class Decoder(nn.Module):
    def __init__(self, batch_size=32):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dense = nn.Linear(128 + 768, 4 * 4 * 512) #Output from dense layer
        self.reshape_dims = (512, 4, 4)

        self.dec_block_6 = nn.ConvTranspose2d(512, 256 * 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_7 = nn.ConvTranspose2d(256 * 4, 256 * 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_1 = nn.ConvTranspose2d(256 * 2, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_3 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_4 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_block_5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)

        self.bn6 = nn.BatchNorm2d(256 * 4)
        self.bn7 = nn.BatchNorm2d(256 * 2)
        self.bn1 = nn.BatchNorm2d(256)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(32)

    def forward(self, z_cond):
        x = self.dense(z_cond)
        x = F.leaky_relu(x)
        x = x.view(-1, *self.reshape_dims)
        x = F.leaky_relu(self.bn6(self.dec_block_6(x)))
        x = F.leaky_relu(self.bn7(self.dec_block_7(x)))
        x = F.leaky_relu(self.bn1(self.dec_block_1(x)))
        x = F.leaky_relu(self.bn2(self.dec_block_2(x)))
        x = F.leaky_relu(self.bn3(self.dec_block_3(x)))
        x = F.leaky_relu(self.bn4(self.dec_block_4(x)))
        x = self.dec_block_5(x)
        return x
        
    
    
class ConvCVAE(nn.Module):
    def __init__(self, encoder,decoder, label_dim, latent_dim, batch_size=32, beta=1, image_dim=[64, 64, 3]):
        super(ConvCVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.label_dim = label_dim
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.beta = beta
        self.image_dim = image_dim

    def __call__(self, inputs, is_train):
        input_image, input_label, conditional_input = self.conditional_input(inputs) #combination of all used as conditional input for the encoder
        enc_output = self.encoder(input_image, input_label, conditional_input, self.latent_dim, is_train)
        z_mean, z_log_var = torch.chunk(enc_output, 2, dim=1)
        print("....done encoding")
        """
        The reparameterization trick is used to sample from the latent space. 
        It generates a sample z_cond from the distribution defined by z_mean and z_log_var. The label (input_label) 
        is concatenated to the latent vector to ensure the decoder can condition on the label.
        This trick is crucial in VAEs to make the sampling process differentiable for backpropagation.
        """
        z_cond = self.reparameterization(z_mean, z_log_var, input_label)
        logits = self.decoder(z_cond, is_train)
        print("....done decoding")
        recon_image=torch.sigmoid(logits)

        #Loss Function: Reconstruction Loss and KL Divergence Loss
        #KL Divergence formula The formula for the KL divergence between two Gaussian distributions is: D KL(Q∣∣P)= 1/2(exp(log(mean^2))+μ^2−1−log(σ^2))
        # 0.5 is used to scale the loss fucntion to fit the ELBO loss
        latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1)
        #Reconstruction loss measures how well the model reconstructs the input image.
        #This loss measures the pixel-wise error between the original and generated images.
        if input_image.shape[-1] != recon_image.shape[-1] or input_image.shape[-2] != recon_image.shape[-2]:
          input_image = F.interpolate(input_image, size=recon_image.shape[-2:], mode="bilinear", align_corners=False)

        reconstr_loss = F.binary_cross_entropy(
          recon_image.view(recon_image.size(0), -1),
          input_image.view(input_image.size(0), -1),
          reduction='none'
          ).sum(dim=1).mean()
        #The role of beta is to control the importance of the KL term relative to the reconstruction loss.
        loss = reconstr_loss + self.beta * latent_loss
        loss = loss.mean() # calculates the average loss over the entire batch

        return {
            'recon_image': recon_image,
            'latent_loss': latent_loss,
            'reconstr_loss': reconstr_loss,
            'loss': loss,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }
    
    def conditional_input(self, inputs):
        # Input layer for image
        input_image = inputs[0]
        # Input layer for the BERT [CLS] token embedding (text embedding)
        input_label = inputs[1]
        #reshaping labels to match the image input dimensions
        labels = input_label.view(-1, self.label_dim, 1, 1) # Shape: (batch_size=32, 1, 1, label_dim)
        # Create a tensor of ones to broadcast the label across the image's spatial dimensions
        labels = labels.expand(-1, -1, self.image_dim[0], self.image_dim[1])
        conditional_input = torch.cat([input_image, labels], dim=1)# Concatenate along the channel axis
        return input_image, input_label, conditional_input

    def reparameterization(self, z_mean, z_log_var, input_labels):
        eps = torch.randn_like(z_mean)
        z = z_mean + torch.exp(0.5 * z_log_var) * eps #making the sample differentiable to apply gradients
        z_cond = torch.cat([z, input_labels], dim=1) #z_cond is latent variable and conditional information in one vector
        return z_cond











        