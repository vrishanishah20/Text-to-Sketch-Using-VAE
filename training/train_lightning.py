import torch
from torch.utils.data import DataLoader, random_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.callbacks import ModelCheckpoint

from ConvolutionalCondVAE import Encoder, Decoder, ConvCVAE
from convcondvae_lightning import LightningConvCVAE 
from preprocessing import SketchDataset
import os

os.environ["PL_DISABLE_FAST_DEV_RUN"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Hyperparameters 
latent_dim = 128
label_dim = 50
beta = 0.5
batch_size = 32
learning_rate = 5e-4
epochs = 300

# Dataset loading
full_dataset = SketchDataset(split="all")
total_size = len(full_dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

torch.manual_seed(42)
train_dataset, val_dataset, test_dataset = random_split(
    full_dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=8,
    drop_last=False, 
    pin_memory=True #helps faster training
)
val_loader = DataLoader(val_dataset, batch_size=batch_size, drop_last=False, pin_memory=True)

encoder = Encoder(latent_dim=latent_dim)
decoder = Decoder(batch_size=batch_size)
base_model = ConvCVAE(encoder, decoder, label_dim=label_dim, latent_dim=latent_dim, beta=beta)

# Applying Pytorch lightning
lightning_model = LightningConvCVAE(base_model, learning_rate=learning_rate)

# checkpoint_callback = ModelCheckpoint(
#     dirpath="/content/drive/MyDrive/VAE_Project/checkpoints/",
#     filename="cvae-epoch={epoch:02d}",
#     every_n_epochs=1,
#     save_top_k=-1,
#     save_weights_only=True
# )

# Logging on tensorboard
logger = TensorBoardLogger("lightning_logs", name="cvae")

#Trainer 
trainer = Trainer(
    max_epochs=epochs,
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    logger=logger,
    # callbacks=[checkpoint_callback], 
    enable_checkpointing=False,
    reload_dataloaders_every_n_epochs=1,
    precision="16-mixed", #Mixed precision applied for faster training
    gradient_clip_val=1.0 
)

print(f"fast_dev_run = {trainer.fast_dev_run}") #this was to check on one datapoint for sanity check, the print statement is to check that it doesnt run otherwise
print(f"Training batches: {len(train_loader)}")

trainer.fit(
    model=lightning_model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)

print(len(train_loader))
