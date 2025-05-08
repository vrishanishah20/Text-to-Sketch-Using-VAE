import torch
import torchvision
import os
import time
import numpy as np
from datetime import datetime
from torch.utils.data import DataLoader
from ConvolutionalCondVAE import ConvCVAE, Decoder, Encoder
from preprocessing import SketchDataset  
from utils import train_step, save_model_checkpoint  
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
latent_dim = 128
label_dim = 50  # Max token length after padding
image_dim = [128, 128, 3]  # Image size (128x128, RGB)
beta = 0.65  # Beta-VAE hyperparameter
epochs = 20
batch_size = 32
learning_rate = 0.001

# Load dataset
train_dataset = SketchDataset(split="train")
val_dataset = SketchDataset(split="val")

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# Initialize model
encoder = Encoder(latent_dim)
decoder = Decoder()
model = ConvCVAE(encoder, decoder, label_dim=label_dim, latent_dim=latent_dim, image_dim=image_dim, beta=beta)

model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Checkpoint path and saving model
checkpoint_root = "/home/jci0365/Text-to-Sketch-Using-VAE/training/checkpoints"
if not os.path.exists(checkpoint_root):
    os.makedirs(checkpoint_root)
checkpoint_prefix = "model"
save_prefix = os.path.join(checkpoint_root, checkpoint_prefix)

# Result folder for saving images and model checkpoints
result_folder = "/home/jci0365/Text-to-Sketch-Using-VAE/training/results"
if not os.path.exists(result_folder):
    os.makedirs(result_folder)

def load_clip_embeddings(image_ids, embeddings_dir="embeddings/"):

    #loading embeddings
    
    embeddings = []
    for image_id in image_ids:
        file_path = "/home/jci0365/Text-to-Sketch-Using-VAE/image_embeddings.npy"

        embedding = np.load(file_path)
        embeddings.append(embedding)
    
    # Convertint list of embeddings to tensor
    return torch.tensor(embeddings, dtype=torch.float32)

def train():
    start_time_total = time.perf_counter()

    for epoch in range(epochs):
        model.train()  
        running_loss = 0.0
        running_recon_loss = 0.0
        running_latent_loss = 0.0
  
        for step_index, (images, labels, clip_embeddings) in enumerate(train_loader):
            
            images, labels, clip_embeddings = images.to(device), labels.to(device), clip_embeddings.to(device)


            optimizer.zero_grad()  # clearing previous graidents from cache

            outputs = model(images, labels, clip_embeddings, is_train=True)

            # Computing the loss
            total_loss, recon_loss, lat_loss = train_step((images, labels, clip_embeddings), model, optimizer)

            total_loss.backward()  # Backpropagating the gradients
            optimizer.step()  # Updateing the weights

            #finding total loss from training
            running_loss += total_loss.item()
            running_recon_loss += recon_loss.item()
            running_latent_loss += lat_loss.item()

            if (step_index + 1) % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{step_index+1}/{len(train_loader)}], "
                        f"Loss: {total_loss.item():.4f}, Recon Loss: {recon_loss.item():.4f}, Latent Loss: {lat_loss.item():.4f}")

        # Saving model and checkpoints
        if (epoch + 1) % 5 == 0:
            save_model_checkpoint(model, save_prefix, epoch)

        print(f"Epoch [{epoch+1}/{epochs}] completed. "
              f"Avg Loss: {running_loss/len(train_loader):.4f}, "
              f"Avg Recon Loss: {running_recon_loss/len(train_loader):.4f}, "
              f"Avg Latent Loss: {running_latent_loss/len(train_loader):.4f}")
        
        # Validation phase after each epoch
        model.eval()  
        with torch.no_grad():
            val_loss = 0.0
            for val_images, val_labels, val_clips in val_loader:
                val_images, val_labels, val_clips = val_images.to(device), val_labels.to(device), val_clips.to(device)
                
                # Forward pass
                val_outputs = model( val_images, val_labels, val_clips, is_train=False )

                # Computing the validation loss
                total_val_loss, recon_val_loss, lat_val_loss = train_step((val_images, val_labels, val_clips), model, optimizer)
                val_loss += total_val_loss.item()

            avg_val_loss = val_loss / len(val_loader)
            print(f"Epoch [{epoch+1}/{epochs}], Validation Loss: {avg_val_loss:.4f}")

    # After training, saves the final model checkpoint
    save_model_checkpoint(model, save_prefix, epochs - 1)
    print("Training completed!")
    # Checking total training time
    exec_time_total = time.perf_counter() - start_time_total
    print(f"Total training time: {exec_time_total:.2f} seconds")

    

# # Function to plot and save reconstructed images
# def plot_reconstructed_images(model, epoch):
#     model.eval()  
#     with torch.no_grad(): 
#         sample_image, sample_label, sample_clip = next(iter(train_loader))
#         sample_image, sample_label, sample_clip = sample_image.to(device), sample_label.to(device), sample_clip.to(device)

#         reconstructed_images = model(sample_image, sample_label, sample_clip, is_train=False)['recon_img']

#         # Plot the original and reconstructed images
#         fig, axes = plt.subplots(1, 2, figsize=(12, 6))
#         axes[0].imshow(convert_batch_to_image_grid(sample_image.cpu()))  
#         axes[0].set_title("Original Images")
#         axes[0].axis('off')

#         axes[1].imshow(convert_batch_to_image_grid(reconstructed_images.cpu()))  
#         axes[1].set_title("Reconstructed Images")
#         axes[1].axis('off')

#         plt.tight_layout()
#         plt.savefig(f"{result_folder}/reconstructed_epoch_{epoch}.png")
#         plt.show()

# # Function to convert batch of images into a grid for visualization
# def convert_batch_to_image_grid(batch_images):
#     # Assuming batch_images are PyTorch tensors of shape (batch_size, 3, height, width)
#     grid = torchvision.utils.make_grid(batch_images, nrow=8, padding=2, normalize=True)
#     return grid.permute(1, 2, 0)  

if __name__ == "__main__":
    train()
