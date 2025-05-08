import torch
from transformers import BertTokenizer, BertModel
from PIL import Image
import pandas as pd
import numpy as np
import os
from datasets import load_dataset
from transformers import CLIPProcessor, CLIPModel

def create_clip_embeddings():
    ds = load_dataset("zoheb/sketch-scene")
    image = ds['train'][0]['image']
    print(type(image)) 
    images = [item['image'] for item in ds['train']] #PIL image objercts
    # image_ids = [item['image'] for item in ds['train']]  # This will give the list of image IDs
    text_labels = [item['text'] for item in ds['train']]  # This will give the corresponding labels

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
    model.to(device)

    embeddings = []

    for idx, (image, label) in enumerate(zip(images, text_labels)):
        print(f"Processing image {idx + 1} of {len(images)}")

        if isinstance(image, Image.Image):  # Ensure we have a PIL Image object
            # Process the image and text using CLIP's processor
            inputs = processor(text=label, images=image, return_tensors="pt", padding=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to the GPU

            # Generate embeddings
            with torch.no_grad():
                outputs = model(**inputs)

            image_embeds = outputs.image_embeds.cpu().detach().numpy()  # Get the image embedding
            embeddings.append(image_embeds)
        else:
            print(f"Invalid image at index {idx}, skipping.")

        #saving embeddings 
        embeddings_array = np.array(embeddings).squeeze(axis=1)
        embedding_file="image_embeddings.npy"
        np.save(embedding_file, embeddings_array)
    
        print(f"Embeddings for {len(images)} images saved to {embedding_file}")
    

create_clip_embeddings()