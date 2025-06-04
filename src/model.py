import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics

class LitCNN(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        self.criterion = nn.CrossEntropyLoss()
        self.acc = torchmetrics.classification.Accuracy(task="multiclass", num_classes=num_classes)
        self.f1 = torchmetrics.classification.F1Score(task="multiclass", num_classes=num_classes)

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    def _step(self, batch, stage):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = self.acc(logits, y)
        f1 = self.f1(logits, y)
        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        self.log(f"{stage}_f1", f1, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        self._step(batch, 'val')
