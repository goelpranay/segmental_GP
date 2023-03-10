import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, WeightedRandomSampler, RandomSampler
import pandas as pd
import os
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import Inception3, Inception_V3_Weights
from torch.nn.functional import softmax, cross_entropy, mse_loss
from torch.optim import Adam
import matplotlib.pyplot as plt
import numpy as np

# Set the precision to medium
torch.set_float32_matmul_precision("medium")

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

# from torch.utils.tensorboard import SummaryWriter

from customdataset import CustomDataset
from fullhand_model_lghtng import ResNet50
from fullhand_model_inception_lghtng import Inception

# define path to the folder and csv file
img_folder_path = "./data/fh_train"
csv_file_path = "./data/train.csv"

transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomAdjustSharpness(sharpness_factor=1.5),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomCrop(size=(500, 500)),
        transforms.Resize((299, 299), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.14, 0.14, 0.14), (0.18, 0.18, 0.18)),
    ]
)

# create instance of custom dataset
full_dataset = CustomDataset(img_folder_path, csv_file_path, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    full_dataset, [train_size, val_size]
)

# # Construct a weighted random sampler
#
# # Extract the labels from the dataset
# labels = np.array([label for _, label in train_dataset])
# labels = np.round(labels).astype(np.int64)
# num_classes = len(np.unique(labels))
#
# # Compute the class weights
# class_counts = np.bincount(labels, minlength=num_classes)  # Add a small constant to avoid divide by zero
# class_weights = 1.0 / class_counts
#
# # Create a mapping from label to index in class_weights
# label_to_idx = {label: i for i, label in enumerate(np.unique(labels))}
#
# # Compute the sample weights
# sample_weights = []
# for label in labels:
#     sample_weights.append(class_weights[label_to_idx[label]])
#
# # Convert class_weights and sample_weights to PyTorch tensors
# class_weights = torch.tensor(class_weights, dtype=torch.float32)
# sample_weights = torch.tensor(sample_weights, dtype=torch.int64)
#
# # Create the WeightedRandomSampler instance
# sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

# create dataloader for the dataset
batch_size = 64
train_loader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=32
)
val_loader = DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False, num_workers=32
)

# Calculate the normalization mean and std
data = next(iter(train_loader))
mean = data[0].mean(dim=(0, 2, 3))  # Calculate mean across each channel
std = data[0].std(dim=(0, 2, 3))  # Calculate std across each channel

print(f"Mean of {batch_size} images:", mean)
print(f"Std of {batch_size} images:", std)

# show an image
for batch_idx, (data, label) in enumerate(train_loader):
    img = data[0]
    img = transforms.ToPILImage()(img)
    img.show()

    # break out of loop after showing
    break

## Lightning training--

# create model
# model = ResNet50()
model = Inception()

# # suggest learning rate
# lr_monitor = LearningRateMonitor(logging_interval='step')
# trainer = pl.Trainer(accelerator='gpu', devices=1, callbacks=[lr_monitor], max_epochs=1, num_sanity_val_steps=0)
# lr_finder = trainer.tuner.lr_find(model, train_loader, val_loader)
# suggested_lr = lr_finder.suggestion()
# print(f"Suggested learning rate: {suggested_lr:.2E}")


# create trainer
trainer = pl.Trainer(
    accelerator="gpu",
    devices=1,
    max_epochs=50,
    reload_dataloaders_every_n_epochs=True,
    logger=pl.loggers.TensorBoardLogger("lightning_logs/", name="regression"),
    log_every_n_steps=25,
    fast_dev_run=False,
)

# specify which metrics to log
# trainer.log_every_n_steps = 10
trainer.log_every_n_epochs = 1
trainer.log_metrics = ["epoch", "loss", "accuracy"]

# train the model
trainer.fit(model, train_loader, val_loader)

# Save the final model weights
torch.save(model.state_dict(), "fullhand_trainedmodel.pt")

# Check predictions on validation data
predictions = trainer.predict(model, dataloaders=val_loader)
predictions = torch.cat(predictions).numpy()
# get the labels from the validation dataset
val_labels = []
for _, labels in val_dataset:
    val_labels.append(labels)
val_array = np.array(val_labels, dtype=np.float32)
plt.scatter(val_array, predictions, label="train subset (val) predictions")
plt.plot([0, 228], [0, 228], color='red')
plt.xlabel('bone age')
plt.ylabel('full hand GP predictions')
plt.legend()
plt.show()

