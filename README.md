# CNN Classification with PyTorch Lightning

This project trains a simple convolutional neural network on images stored in the `Images` folder. Labels are provided in `Label.csv`.

## Requirements

- Python 3.8+
- torch
- torchvision
- pytorch-lightning
- torchmetrics
- pandas
- scikit-learn
- pillow

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Training

Run the training script which uses batch size of 64 and trains for 30 epochs:

```bash
python train.py
```

The script logs Accuracy and F1 score on the validation set.
