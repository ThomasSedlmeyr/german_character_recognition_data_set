import os

import pandas as pd
import numpy as np
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OneHotEncoder

class GermanCharacterRecognitionDS(Dataset):
    def __init__(self, path_csv, one_hot_encoder, transform=None, target_transform=None, classes=[],
                 num_channels=1):
        self.path_csv = path_csv
        self.transform = transform
        self.target_transform = target_transform
        self.data_lines = self.read_lines_csv(classes)
        self.n = len(self.data_lines)
        self.classes = classes
        self.onehot_encoder = one_hot_encoder
        self.num_channels = num_channels

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label, image = self.parse_one_line(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        label = self.onehot_encoder.transform(np.array(label).reshape(-1, 1))[0]
        return image, label

    def read_lines_csv(self, classes):
        training_data_file = open(self.path_csv, 'r', encoding="latin-1")
        data_lines = training_data_file.readlines()
        training_data_file.close()
        data_lines = [line for line in data_lines if line[0] in classes]
        return data_lines

    def parse_one_line(self, index):
        line = self.data_lines[index].split(',')
        image_np = np.asarray(line[1:1601], dtype=np.float32)
        image_np = image_np.reshape(40, 40, 1)
        if self.num_channels:
            image_np = np.repeat(image_np, self.num_channels, axis=2)
        # image_tensor = torch.from_numpy(image_np)
        return line[0], image_np
