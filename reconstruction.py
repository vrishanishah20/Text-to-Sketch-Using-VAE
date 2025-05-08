import pickle
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel
from PIL import Image
from ConvolutionalCondVAE import Encoder, Decoder, ConvCVAE
from convcondvae_lightning import LightningConvCVAE
import os
import numpy as np

latent_dim = 128
label_dim = 50
beta = 0.1
batch_size = 16
ckpt_path = "/content/drive/MyDrive/VAE_Project/cvae-epoch20.ckpt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# pickled dataset
with open("/content/drive/MyDrive/VAE_Project/clean_test_set.pkl", "rb") as f: #had to reload the test set 
    test_data = pickle.load(f)

# tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

clip_embeddings_np = np.load("/content/drive/MyDrive/VAE_Project/image_embeddings.npy")

#transfomration, same as train
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# using pickled datset wrapper
class PickledSketchDataset(Dataset):
    def __init__(self, data, offset=0):
        self.data = data
        self.offset = offset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = transform(sample["image"].convert("RGB"))
        text = sample["text"]
        clip_embed = torch.tensor(clip_embeddings_np[self.offset + idx], dtype=torch.float32)

        # Text embedding
        tokens = tokenizer(text, padding="max_length", truncation=True, max_length=50, return_tensors="pt")
        with torch.no_grad():
            bert_out = bert(**{k: v.to(device) for k, v in tokens.items()})
        text_embed = bert_out.last_hidden_state[:, 0, :].squeeze(0)

        return image, text_embed, clip_embed

test_dataset = PickledSketchDataset(test_data, offset=8998)  
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# model
encoder = Encoder(latent_dim=latent_dim)
decoder = Decoder()
base_model = ConvCVAE(encoder, decoder, label_dim=label_dim, latent_dim=latent_dim, beta=beta)
model = LightningConvCVAE.load_from_checkpoint(ckpt_path, base_model=base_model, learning_rate=1e-3).to(device)
model.eval()

model.model.eval()

# currently checking on 5 images
model.model.eval()
model.eval()

num_samples = 5
subset = [test_dataset[i] for i in range(num_samples)]
images, text_embeds, clip_embeds = zip(*subset)

images = torch.stack(images).to(device)
text_embeds = torch.stack(text_embeds).to(device)
clip_embeds = torch.stack(clip_embeds).to(device)

with torch.no_grad():
    enc_out = model.model.encoder(images, text_embeds, clip_embeds, latent_dim)
    z_mean, z_log_var = torch.chunk(enc_out, 2, dim=1)
    z = model.model.reparameterization(z_mean, z_log_var, text_embeds)
    logits = model.model.decoder(z)
    recon_images = torch.sigmoid(logits)

images = (images + 1) / 2
recon_images = recon_images.clamp(0, 1)


# Undoing normalization to match input
images = (images + 1) / 2
recon_images = recon_images.clamp(0, 1)

def plot_5_recon(orig, recon, filename="reconstruction_5.png"):
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
    for i in range(num_samples):
        axes[0, i].imshow(orig[i].permute(1, 2, 0).cpu())
        axes[0, i].axis('off')
        axes[1, i].imshow(recon[i].permute(1, 2, 0).cpu())
        axes[1, i].axis('off')
    axes[0, 0].set_ylabel("Original", fontsize=12)
    axes[1, 0].set_ylabel("Reconstructed", fontsize=12)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved 5-image reconstruction to {filename}")
    plt.show()

plot_5_recon(images, recon_images)
