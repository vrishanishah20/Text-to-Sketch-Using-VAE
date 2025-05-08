import app as st
import torch
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel
from ConvolutionalCondVAE import Encoder, Decoder, ConvCVAE
from convcondvae_lightning import LightningConvCVAE
from PIL import Image
import io

latent_dim = 128
beta = 0.1
ckpt_path = "cvae-epoch20.ckpt"  # Place the .ckpt file in the same directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model and components
@st.cache_resource
def load_model():
    encoder = Encoder(latent_dim=latent_dim)
    decoder = Decoder()
    base_model = ConvCVAE(encoder, decoder, label_dim=768, latent_dim=latent_dim, beta=beta)
    model = LightningConvCVAE.load_from_checkpoint(ckpt_path, base_model=base_model, learning_rate=1e-3).to(device)
    model.eval()
    return model

@st.cache_resource
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    bert.eval()
    return tokenizer, bert

model = load_model()
tokenizer, bert = load_bert()

# GUI making
st.title("🎨 Text-to-Sketch Generator using CVAE")
user_prompt = st.text_input("Enter a description (e.g., 'a zebra in a field')")

if st.button("Generate Sketch") and user_prompt:
    # Tokenize and encode text
    tokens = tokenizer(user_prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=50)
    with torch.no_grad():
        output = bert(**{k: v.to(device) for k, v in tokens.items()})
    text_embedding = output.last_hidden_state[:, 0, :]

    # Sample z
    z = torch.randn(1, latent_dim).to(device)
    z_cond = torch.cat([z, text_embedding], dim=1)

    # Generating image
    with torch.no_grad():
        logits = model.model.decoder(z_cond)
        generated_img = torch.sigmoid(logits).squeeze(0).cpu()

    # Convert to PIL to show
    img = generated_img.permute(1, 2, 0).numpy()
    img = (img * 255).astype("uint8")
    img_pil = Image.fromarray(img)

    st.image(img_pil, caption="Generated Sketch", use_column_width=True)

