import os

import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader


class CustomImageDataset(Dataset):
    def __init__(self, path_csv, transform=None, target_transform=None):
        self.path_csv = path_csv
        self.transform = transform
        self.target_transform = target_transform
        self.data_lines = self.read_lines_csv()
        self.n = len(self.data_lines)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image, label = self.parse_one_line(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def read_lines_csv(self):
        training_data_file = open(self.path_csv, 'r', encoding="latin-1")
        data_lines = training_data_file.readlines()
        training_data_file.close()
        return data_lines

    def parse_one_line(self, index):
        line = self.data_lines[index].split(',')
        image_np = np.asarray(line[1:1601])
        # image_tensor = torch.from_numpy(image_np)
        return line[0], image_np
