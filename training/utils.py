import torch

#def save_model_checkpoint(model, save_prefix, epoch):

    #Save the model checkpoint at a given epoch.
    
    #checkpoint = {'model_state_dict': model.state_dict(), 'epoch': epoch}
    #torch.save(checkpoint, f"{save_prefix}_epoch_{epoch}.pth")
    #print(f"Checkpoint saved at epoch {epoch}.")

def train_step(inputs, model, clip_embeddings, optimizer):
    #Performs one training step for thge forward pass, loss computation, backward pass, and weights
    
    images, labels = inputs
    optimizer.zero_grad()  

    # Forward pass
    outputs = model(images, labels, clip_embeddings, is_train=True)

    # Get image embeddings and reconstructed image
    recon_img = outputs['recon_img']
    image_embeds = outputs['image_embeds']
    
    # reconstructino loss
    recon_loss = F.binary_cross_entropy_with_logits(
        recon_img.view(-1, 64 * 64 * 3), 
        images.view(-1, 64 * 64 * 3)
    )

    # KL Divergence Loss for labels to understand the meana nd std dev with the conditions, the beta is a factor on which loss to give more importance to
    z_mean = outputs['z_mean']
    z_log_var = outputs['z_log_var']
    latent_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp(), dim=-1).mean()

    # Compute CLIP Embedding Loss 
    clip_loss = F.mse_loss(image_embeds, clip_embeddings)

    # Total loss is weighted ELBO loss
    total_loss = recon_loss + latent_loss + clip_loss

    # Backpropagation
    total_loss.backward()  
    optimizer.step()  

    return total_loss, recon_loss, latent_loss, clip_loss