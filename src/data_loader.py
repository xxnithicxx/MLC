import random
import pandas as pd
import os
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from src.config import CONFIG

class MultiLabelDataset(Dataset):
    def __init__(self, data, image_dir, transform=None):
        self.data = data
        self.image_dir = image_dir
        self.transform = transform
        self.classes = data.columns[2:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image, labels = self.load_image_and_labels(row)
        if self.transform:
            image = self.transform(image)
        label_tensor = torch.tensor([int(lbl) for lbl in labels], dtype=torch.float)
        return image, label_tensor

    def load_image_and_labels(self, row):
        image_path = os.path.join(self.image_dir, row['Image_Name'])
        image = Image.open(image_path).convert('RGB')
        labels = row.iloc[-10:].tolist()
        return image, labels

def load_data(csv_path):
    data = pd.read_csv(csv_path)
    columns_names = data.columns
    data.rename(columns={columns_names[1]: 'Classes'}, inplace=True)
    return data

def get_data_loaders(mode="full"):
    """
    mode: "full" for the entire dataset, "limited" for 10 random samples
    """
    # Load dataset
    data = load_data(CONFIG['csv_file'])

    # Split into train and validation sets
    train_data, val_data = train_test_split(data, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'])

    if mode == "limited":
        # Select 10 random samples from the train data for quick inference
        indices = random.sample(range(len(train_data)), 10)
        train_data = train_data.iloc[indices]

    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(CONFIG['image_size']),
        transforms.ToTensor()
    ])

    # Create dataset instances
    train_dataset = MultiLabelDataset(train_data, CONFIG['images_dir'], transform=transform)
    val_dataset = MultiLabelDataset(val_data, CONFIG['images_dir'], transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

    return train_loader, val_loader