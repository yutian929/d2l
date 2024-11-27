import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class BananaDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with annotations.
            img_dir (str): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.annotations.iloc[idx, 1]
        bbox = self.annotations.iloc[idx, 2:].values.astype("float32")  # xmin, ymin, xmax, ymax

        if self.transform:
            image = self.transform(image)

        target = {
            "label": torch.tensor(label, dtype=torch.long),
            "bbox": torch.tensor(bbox, dtype=torch.float32),
        }

        return image, target
