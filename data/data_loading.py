from datasets import load_dataset

# Load your dataset
ds = load_dataset("zoheb/sketch-scene")

# Check the keys of the dataset (e.g., train, test, etc.)
print(ds.keys())

# Check a sample from the 'train' split
print(ds['train'][0])

# Check the first item in the training dataset
sample = ds['train'][0]
print(sample)
