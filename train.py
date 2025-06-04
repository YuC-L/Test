import pytorch_lightning as pl
from pytorch_lightning import Trainer
from src.datamodule import ImageDataModule
from src.model import LitCNN


def main():
    data = ImageDataModule(csv_file='Label.csv', img_dir='Images', batch_size=64)
    data.setup()
    model = LitCNN(num_classes=data.num_classes)
    trainer = Trainer(max_epochs=30)
    trainer.fit(model, datamodule=data)

if __name__ == '__main__':
    main()
