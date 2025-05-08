from datasets import load_dataset
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from transformers import BertTokenizer, BertModel
from PIL import Image
import pickle
import numpy as np

## Loading Dataset from HuggingFace

class SketchDataset(Dataset):
    def __init__(self, learning_rate = 0.001, train_size=0.8, split="train", validation_size=0.1, batch_size=32, save_test_set=True):
        #Using datasets library
        self.ds = load_dataset("zoheb/sketch-scene")

        #dataset size
        total_len = len(self.ds['train'])
        print("Total dataset size:", total_len)
        #tokenizer for labels
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        #BERT model for generating text embeddings
        self.bert_model = BertModel.from_pretrained("bert-base-uncased")

        self.transform = transforms.Compose([
            transforms.Resize((128,128)),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.5,0.5,0.5], std = [0.5,0.5,0.5]) #normalizing from [-1,1] for reconstr loss
        ])
        torch.manual_seed(42)
        train_len = int(total_len*train_size)
        validation_len = int(total_len*validation_size)
        # test_len = int(total_len*validation_size)
        test_len = total_len - train_len - validation_len
        print(f"Train: {train_len}, Val: {validation_len}, Test: {test_len}")
        
        self.offset = 0
        if split == "train":
            self.data = self.ds['train'].select(range(train_len))
            self.offset = 0
        elif split == "val":
            self.data = self.ds['train'].select(range(train_len, train_len + validation_len))
            self.offset = train_len
        elif split == "test":
            self.data = self.ds['train'].select(range(train_len + validation_len, total_len))
            self.offset = train_len + validation_len
        else:  
            self.data = self.ds['train']
            self.offset = 0
        # if split == "train":
        #     self.data = self.ds['train'][:train_len]
        #     self.offset = 0
        # elif split == "val":
        #     self.data = self.ds['train'][train_len:train_len + validation_len]
        #     self.offset = train_len
        # else:
        #     self.data = self.ds['train'][train_len + validation_len:]
        #     self.offset = train_len + validation_len
            if save_test_set:
                with open("test_set.pkl", "wb") as f:
                    pickle.dump(self.data, f)
        
        self.clip_embeddings = np.load("/home/jci0365/Text-to-Sketch-Using-VAE/image_embeddings.npy")


    def text_tokenizing(self, text):
        tokens = self.tokenizer(text, truncation=True, padding='max_length', max_length=50, return_tensors='pt')
        return tokens['input_ids'].squeeze(0), tokens['attention_mask'].squeeze(0) #attention mask telling us which tokens are real and which are padding
    
    def text_embeddings(self, text):
        #[CLS] token's representation as the label embedding.
        input_ids, attention_masks = self.text_tokenizing(text)
        with torch.no_grad():
            outputs = self.bert_model(input_ids.unsqueeze(0), attention_mask=attention_masks.unsqueeze(0))

        cls_embedding = outputs.last_hidden_state[:, 0, :]  # shape: (embedding_dim,) #captures the meaning of the entire sequence
        
        return cls_embedding.squeeze(0)


    def preprocess_image(self, image):
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = image.resize((128, 128))  
        image_tensor = transforms.ToTensor()(image)  # Converts to [0, 1] float tensor
        return image_tensor
    
  
    
    def __getitem__(self, idx):
      sample = self.data[idx]  
      if self.transform:
          image_tensor = self.transform(sample['image'])
      else:
          image_tensor = transforms.ToTensor()(sample['image']) 

      text_tensor = self.text_embeddings(sample['text'])
      clip_tensor = torch.tensor(self.clip_embeddings[self.offset + idx], dtype=torch.float32)

      return image_tensor, text_tensor, clip_tensor
    def __len__(self):
        return len(self.data)
        
##Creating dataLoader for mini batches in training

# Now, create separate DataLoaders for training and validation data
# train_dataset = SketchDataset()
# val_dataset = SketchDataset()

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)


