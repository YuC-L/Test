from pathlib import Path
from sklearn.model_selection import train_test_split
import pandas as pd
from torch.utils.data import DataLoader
import torchvision.transforms as T
import pytorch_lightning as pl

from .dataset import ImageDataset

class ImageDataModule(pl.LightningDataModule):
    def __init__(self, csv_file, img_dir, batch_size=64):
        super().__init__()
        self.csv_file = csv_file
        self.img_dir = Path(img_dir)
        self.batch_size = batch_size
        self.transform = T.Compose([
            T.Resize((64, 64)),
            T.ToTensor(),
        ])

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        df = pd.read_csv(self.csv_file)
        labels = sorted(df['Label'].unique())
        self.label2idx = {l: i for i, l in enumerate(labels)}
        df['LabelIdx'] = df['Label'].map(self.label2idx)
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['LabelIdx'])
        self.train_ds = ImageDataset(train_df, self.img_dir, self.transform)
        self.val_ds = ImageDataset(val_df, self.img_dir, self.transform)
        self.num_classes = len(labels)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size)
