import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os


class SegmentalDataset(Dataset):
    def __init__(self, fh_img_folder_path, wrist_img_folder_path, csv_file_path, fh_transform=None, wrist_transform=None):
        self.fh_img_folder_path = fh_img_folder_path
        self.wrist_img_folder_path = wrist_img_folder_path
        self.df = pd.read_csv(csv_file_path)
        self.fh_transform = fh_transform
        self.wrist_transform = wrist_transform


        # filter out by sex
        self.df = self.df[self.df["male"] == True]  # network applicable only to boys

        # filter out entries without corresponding images
        # BUG -- ToDo: doesn't check if same files are in both folders --
        self.df = self.df[
            self.df["id"].apply(
                lambda x: os.path.exists(os.path.join(wrist_img_folder_path, str(x) + ".png"))
            )
        ]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        fh_img_name = self.df.iloc[idx]["id"]
        fh_img_path = os.path.join(self.fh_img_folder_path, str(fh_img_name) + ".png")
        wrist_img_name = self.df.iloc[idx]["id"]
        wrist_img_path = os.path.join(self.wrist_img_folder_path, str(wrist_img_name) + ".png")
        label = self.df.iloc[idx]["boneage"]

        fh_img = Image.open(fh_img_path).convert("RGB")
        wrist_img = Image.open(wrist_img_path).convert("RGB")

        if self.transform:
            fh_img = self.fh_transform(fh_img)
            wrist_img = self.wrist_transform(wrist_img)

        return fh_img, wrist_img, label
