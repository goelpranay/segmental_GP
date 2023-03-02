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
import csv

# Set the precision to medium
torch.set_float32_matmul_precision("high")

import pytorch_lightning as pl

from pytorch_lightning import Trainer

# maybe the resnets are different -- check --
from fullhand_model_lghtng import ResNet50
from wrist_model_lghtng import ResNetReg


# load the trained models
fh_trained_model = ResNet50()
fh_trained_model.load_state_dict(torch.load("fullhand_trainedmodel_10Kepochs.pt"))

wrist_trained_model = ResNetReg()
wrist_trained_model.load_state_dict(torch.load("wrist_trainedmodel_10Kepochs.pt"))
trainer = Trainer(accelerator="gpu", devices=1)

fh_preds = []
wrist_preds = []
labels = []

with open("./data/val.csv") as csvfile:
    reader = csv.reader(csvfile)
    next(reader) # skip the header

    # Enumerate through the rows
    for i, row in enumerate(reader):

        if i >= 25:
            break  # stop after some rows

        if row[2] == "True":  # male = True
            # Use indexing to access individual elements of the row
            print(f"Row {i}: {row[0]}, {row[1]}")

            try:
                img = Image.open("./data/wrist_val/" + row[0] + ".png").convert("RGB")
                # print(f"{img.size=}")
            except FileNotFoundError:
                # Handle the error if the file is not found
                print("Error: file not found")
            else:
                # Preprocess the wrist image
                transform = transforms.Compose(
                    [
                        transforms.RandomHorizontalFlip(0), # for debug reasons
                        transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            (0.3659, 0.3659, 0.3659), (0.2857, 0.2857, 0.2857)
                        ),
                    ]
                )
                img = transform(img)

                # Make a wrist prediction
                with torch.no_grad():
                    prediction = wrist_trained_model(img.unsqueeze(0))
                    print(f"Wrist prediction: {prediction.item()}")

                    # Create a PyTorch DataLoader that contains the input image
                    im_label = torch.tensor([1.])  # Dummy label for the loader
                    im_label = im_label.unsqueeze(dim=0)  # Add a batch dimension to the label
                    loader = torch.utils.data.DataLoader(
                        [(img, im_label)], batch_size=1, shuffle=False
                    )
                    # Use the trainer to make a prediction on the input image
                    tr_pred = trainer.predict(wrist_trained_model, loader)
                    print(f"Wrist; trainer prediction: {tr_pred}")

                    boneage = float(row[1])
                    # wrist_preds.append(prediction.item())
                    wrist_preds.append((tr_pred[0].squeeze()+prediction.item())/2)
                    labels.append(boneage)



                    # Preprocess the full hand image
                    img = Image.open("./data/fh_val/" + row[0] + ".png").convert("RGB")
                    transform = transforms.Compose(
                        [
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize((0.14, 0.14, 0.14), (0.18, 0.18, 0.18)),
                        ]
                    )
                    img = transform(img)

                    # Make a prediction
                    with torch.no_grad():
                        prediction = fh_trained_model(img.unsqueeze(0))
                        print(f"Full hand prediction: {prediction.item()}")

                        # Create a PyTorch DataLoader that contains the input image
                        im_label = torch.tensor([1.])  # Dummy label for the loader
                        im_label = im_label.unsqueeze(dim=0)  # Add a batch dimension to the label
                        loader = torch.utils.data.DataLoader(
                            [(img, im_label)], batch_size=1, shuffle=False
                        )
                        # Use the trainer to make a prediction on the input image
                        tr_pred = trainer.predict(fh_trained_model, loader)
                        print(f"Full hand; trainer prediction: {tr_pred}")

                        fh_preds.append((tr_pred[0].squeeze() + prediction.item()) / 2)


        else:
            print(f"Male? is False")


# the prediction stacks
wrist_preds_array = np.stack([tensor.numpy() for tensor in wrist_preds])
print(wrist_preds_array)
print(labels)

fh_preds_array = np.stack(fh_preds)
print(fh_preds_array)

# average the wrist and full hand predictions
avg_preds_array = np.mean([fh_preds_array,wrist_preds_array], axis=0)

# compute MADs
MAD_wrist=np.mean(np.abs(labels-wrist_preds_array))
MAD_fh=np.mean(np.abs(labels-fh_preds_array))
MAD=np.mean(np.abs(labels-avg_preds_array))
print(f"{MAD_fh=}")
print(f"{MAD_wrist=}")
print(f"{MAD=}")

# plot the data
plt.scatter(labels, wrist_preds_array, label='wrist predictions')
plt.scatter(labels, fh_preds_array, label='full hand predictions')
plt.scatter(labels, avg_preds_array, label='averaged predictions')
# add a line to the plot
plt.plot([0, 224], [0, 224], color='red')
plt.xlabel('bone age')
plt.legend()
plt.show()
