from torch.utils.data import Dataset
import numpy as np

class GermanCharacterRecognitionDS(Dataset):
    def __init__(self, path_csv, dict_classes_to_numbers, transform=None, target_transform=None, classes=[],
                 num_channels=1):
        self.path_csv = path_csv
        self.transform = transform
        self.target_transform = target_transform
        self.data_lines = self.read_lines_csv(classes)
        self.n = len(self.data_lines)
        self.classes = classes
        self.num_channels = num_channels
        self.dict_classes_to_numbers = dict_classes_to_numbers

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        label, image = self.parse_one_line(idx)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)            
        # We have to convert the label to an integer value
        label = self.dict_classes_to_numbers[label]
        return image, label

    def read_lines_csv(self, classes):
        training_data_file = open(self.path_csv, 'r')
        data_lines = training_data_file.readlines()
        training_data_file.close()
        data_lines = [line for line in data_lines if line[0] in classes]
        return data_lines

    def parse_one_line(self, index):
        line = self.data_lines[index].split(',')
        image_np = np.asarray(line[1:1601], dtype=np.float32)
        image_np = image_np.reshape(40, 40, 1)
        if self.num_channels != 1:
            image_np = np.repeat(image_np, self.num_channels, axis=2)
        return line[0], image_np