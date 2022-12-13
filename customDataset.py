# This file creates a custom dataset with 
# binary labels
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage import io
import cv2 

class FerrisAndBearsDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir 
        self.transform = transform
        self.img_dim = (64, 64)

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 0])
        image = io.imread(img_path)
        image = cv2.resize(image, self.img_dim)

        y_label = torch.tensor(int(self.annotations.iloc[index, 1]))

        if self.transform:
            image = self.transform(image)
        return (image, y_label)