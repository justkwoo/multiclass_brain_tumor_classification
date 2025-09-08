import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class Dataset(Dataset):
    def __init__(self, images: list, labels: list, transform=None):
        super().__init__()
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self, ):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(self.images[index])
        image = image.convert('RGB') #forces RGB
        
        image = self.transform(image)
        label = self.labels[index]

        return image, label