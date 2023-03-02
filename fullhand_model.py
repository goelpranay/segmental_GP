import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
import os
from torchvision import models
from torchvision.models import resnet50, ResNet50_Weights
from torch.nn.functional import softmax, cross_entropy, mse_loss
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np

from customdataset import CustomDataset

print(torchvision.__version__)

# define path to the folder and csv file
img_folder_path = "./full_hand"
csv_file_path = "./train.csv"

# define transformations to be applied to images
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=(-20, 20)),
        transforms.RandomCrop(size=(200, 200)),
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

# create dataloader for the dataset
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

# Calculate the normalization mean and std
data = next(iter(train_loader))
mean = data[0].mean(dim=(0, 2, 3))  # Calculate mean across each channel
std = data[0].std(dim=(0, 2, 3))  # Calculate std across each channel

print(f"Mean of {batch_size} images:", mean)
print(f"Std of {batch_size} images:", std)

# show a batch of images
for batch_idx, (data, label) in enumerate(train_loader):
    # data shape: (batch_size, channels, height, width)
    # label shape: (batch_size, )

    # show the first image in the batch
    # print(data[0].dtype)
    # print(data[1].dtype)
    img = data[0]
    img = transforms.ToPILImage()(img)
    img.show()

    # break out of loop after showing first batch of images
    break


# training the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"{device=}")

model = resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 1)
print(model)
model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

N_EPOCHS = 10

train_loss = []
val_loss = []

for epoch in range(N_EPOCHS):
    print(f"Epoch {epoch+1}/{N_EPOCHS}")

    model.train()

    batch_losses = []

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        y_ = model(x)
        # print(y.dtype) # y.dtype is torch.float64
        # print(y_.size) # y_.dtype is float32
        # y.float().dtype is float32
        loss = mse_loss(y_.squeeze(), y.float())

        loss.backward()
        optimizer.step()

        # Print batch loss
        batch_loss = loss.item()
        print(f"    Batch loss: {batch_loss:.4f}")
        batch_losses.append(batch_loss)

    # After each epoch, print the average batch loss and plot the batch loss values
    avg_batch_loss = sum(batch_losses) / len(batch_losses)
    print(f"Epoch {epoch+1}: average batch loss (train) = {avg_batch_loss:.2f}")
    train_loss.append(avg_batch_loss)
    # batch_losses = []

    # Eval mode
    model.eval()

    batch_losses = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            y_ = model(x)
            loss = mse_loss(y_.squeeze(), y.float())

            # Print batch loss
            batch_loss = loss.item()
            print(f"    Batch loss: {batch_loss:.4f}")
            batch_losses.append(batch_loss)

    # After each epoch, print the average batch loss and plot the batch loss values
    avg_batch_loss = sum(batch_losses) / len(batch_losses)
    print(f"Epoch {epoch + 1}: average batch loss (val) = {avg_batch_loss:.2f}")
    val_loss.append(avg_batch_loss)
    # batch_losses = []

plt.plot(train_loss, color="blue", label="training loss")
plt.plot(val_loss, color="red", label="validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
