import pandas as pd
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset

class ImageDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.img_dir / row['ImageName']
        image = Image.open(img_path).convert('RGB')
        label = row['LabelIdx']
        if self.transform:
            image = self.transform(image)
        return image, label
