import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from ConvolutionalCondVAE import Encoder, Decoder, ConvCVAE
from convcondvae_lightning import LightningConvCVAE

latent_dim = 128
beta = 0.1
ckpt_path = "/content/drive/MyDrive/VAE_Project/cvae-epoch20.ckpt"  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
prompt = "a train on the tracks" 

# toekinzer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert = BertModel.from_pretrained("bert-base-uncased").to(device)
bert.eval()

tokens = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
with torch.no_grad():
    output = bert(**{k: v.to(device) for k, v in tokens.items()})
text_embedding = output.last_hidden_state[:, 0, :]  # CLS token from bert embeddings

# one sample z
z = torch.randn(1, latent_dim).to(device)

# Concatenate latent vector and text embedding 
z_cond = torch.cat([z, text_embedding], dim=1)

# model
encoder = Encoder(latent_dim=latent_dim)
decoder = Decoder()
base_model = ConvCVAE(encoder, decoder, label_dim=768, latent_dim=latent_dim, beta=beta)
model = LightningConvCVAE.load_from_checkpoint(ckpt_path, base_model=base_model, learning_rate=1e-3).to(device)
model.eval()

# generating images
with torch.no_grad():
    logits = model.model.decoder(z_cond)
    generated_img = torch.sigmoid(logits).squeeze(0).cpu()


plt.imshow(generated_img.permute(1, 2, 0))
plt.title(f"Generated Sketch: \"{prompt}\"")
plt.axis('off')
plt.savefig("generated_sketch.png")
plt.show()
