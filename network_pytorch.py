import torch
import torch.nn as nn
import torch.nn.functional as F

class PyTorchClassifier(nn.Module):
    def __init__(self, num_classes):
        super(PyTorchClassifier, self).__init__()
        self.size_fc1 = 256
        self.conv1 = nn.Conv2d(1, 32, 6)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.fc1 = nn.Linear(self.size_fc1, 256)
        #self.fc2 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#class MNISTResnetmodel(nn.Module):
#    def
