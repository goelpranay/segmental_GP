import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, img_folder_path, csv_file_path, transform=None):
        self.img_folder_path = img_folder_path
        self.df = pd.read_csv(csv_file_path)
        # self.df = pd.read_csv(csv_file_path, dtype={'column1': str, 'column2': int, 'column3': bool})
        self.transform = transform

        # filter out by sex
        self.df = self.df[self.df["male"] == True]  # network applicable only to boys

        # filter out entries without corresponding images
        self.df = self.df[
            self.df["id"].apply(
                lambda x: os.path.exists(os.path.join(img_folder_path, str(x) + ".png"))
            )
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx]["id"]
        img_path = os.path.join(self.img_folder_path, str(img_name) + ".png")
        label = self.df.iloc[idx]["boneage"]
        label = label.astype(np.float32)
        # label.dtype

        img = Image.open(img_path).convert("RGB")
        # tensor_img = torch.Tensor(img)

        if self.transform:
            img = self.transform(img)

        return img, label
