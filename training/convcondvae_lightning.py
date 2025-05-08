import pytorch_lightning as pl
import torch
import torch.nn.functional as F

class LightningConvCVAE(pl.LightningModule):
    def __init__(self, base_model, learning_rate=1e-3):
        super().__init__()
        self.model = base_model
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["base_model"])

    def step(self, batch):
      input_image, input_label, conditional_input = batch
      enc_output = self.model.encoder(input_image, input_label, conditional_input, self.model.latent_dim)
      z_mean, z_log_var = torch.chunk(enc_output, 2, dim=1)
      z_cond = self.model.reparameterization(z_mean, z_log_var, input_label)
      logits = self.model.decoder(z_cond)
      # recon_image = torch.sigmoid(logits)
      #Resize input image 
      if input_image.shape[-1] != logits.shape[-1] or input_image.shape[-2] != logits.shape[-2]:
        input_image = F.interpolate(input_image, size=logits.shape[-2:], mode="bilinear", align_corners=False)

      input_image = (input_image + 1) / 2
      # assert recon_image.shape == input_image.shape
      # assert input_image.numel() > 0
      # print("Input image min/max:", input_image.min().item(), input_image.max().item())
      # input_image = input_image.clamp(0, 1)
      # print("Input image range after clamp:", input_image.min().item(), input_image.max().item())
      # recon_image = recon_image.clamp(0, 1)
      # reconstr_loss = F.binary_cross_entropy(
      #     recon_image.view(recon_image.size(0), -1),
      #     input_image.view(input_image.size(0), -1),
      #     reduction='none'
      # ).sum(dim=1).mean()

      reconstr_loss = F.binary_cross_entropy_with_logits(
        logits.view(logits.size(0), -1),
        input_image.view(input_image.size(0), -1),
        reduction='none'
        ).sum(dim=1).mean()

      latent_loss = -0.5 * torch.sum(
          1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=1
      ).mean()
      
      total_loss = reconstr_loss + self.model.beta * latent_loss
      # print(f"Processing batch {batch_idx} of size {len(batch[0])}")
      recon_image = torch.sigmoid(logits) 
      return total_loss, reconstr_loss, latent_loss


    def training_step(self, batch, batch_idx):
        total_loss, recon_loss, latent_loss = self.step(batch)
        self.log("train_loss", total_loss, prog_bar=True)
        self.log("train_reconstr_loss", recon_loss)
        self.log("train_latent_loss", latent_loss)
        return total_loss

    def validation_step(self, batch, batch_idx):
        total_loss, recon_loss, latent_loss = self.step(batch)
        self.log("val_loss", total_loss, prog_bar=True)
        self.log("val_reconstr_loss", recon_loss)
        self.log("val_latent_loss", latent_loss)
        return total_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
