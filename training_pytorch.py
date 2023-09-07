import copy
import torch
import time


import numpy as np
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import torch.nn as nn
from tqdm import tqdm

from data_pytorch import GermanCharacterRecognitionDS
import network_pytorch
from train_utils import EpochInformation, EarlyStopper

from torchvision import  models
from torchvision.models import ResNet18_Weights


path_train_csv = "train.csv"
path_test_csv = "test.csv"
batch_size = 16
num_workers = 12
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
#classes = ['!','$','&','(',')','+','0','1','2','3','4','5','6','7','8','9','<','>','?','A','B','C','D','E','F','G','H',
#           'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i',
#           'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','~','ß','α','β','π','φ','€','∑','√','∞',
#           '∫']
num_classes = len(classes)
hparams = {"num_epochs": 100, "early_stopping_patience": 5, "early_stopping_threshold": 0.001}
onehot_encoder = OneHotEncoder(sparse=False)
reshaped_classes = np.array(classes).reshape(-1, 1)
onehot_encoder = onehot_encoder.fit(reshaped_classes)
def get_train_and_val_loader(train_data_set, num_samples_validation_data=250, onehot_encoder=onehot_encoder,
                             classes=classes):
    # Define the ratio for train and validation data
    print("Splitting train- and val-data ...")
    val_count = dict(zip(classes, len(classes) * [0]))
    val_indices = []
    train_indices = []
    for i in range(len(train_data_set)):
        _, label = train_data_set[i]
        string_label_list = onehot_encoder.inverse_transform(np.reshape(label, (1, -1)))
        label_string = str(string_label_list[0][0])
        number = val_count[label_string]
        if val_count[label_string] < num_samples_validation_data:
            val_count[label_string] += 1
            val_indices.append(i)
        else:
            train_indices.append(i)

    np.save("val_indices_whole_ds.npy", np.asarray(val_indices))
    np.save("train_indices_whole_ds.npy", np.asarray(train_indices))
    train_loader, val_loader = split_train_loader(train_data_set, train_indices, val_indices)
    print("Splitting done")
    return train_loader, val_loader

def split_train_loader(train_data_set, train_indices, val_indices):
    train_loader = torch.utils.data.DataLoader(Subset(train_data_set, train_indices), batch_size=batch_size,
                                               shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(Subset(train_data_set, val_indices), batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)
    return train_loader, val_loader

def get_all_labels_of_data_loader(data_loader, onehot_encoder):
    labels_count_dict = dict(zip(classes, len(classes) * [0]))
    for _, labels in data_loader:
        string_label = onehot_encoder.inverse_transform(labels)
        for label in string_label:
            labels_count_dict[label[0]] += 1
    return labels_count_dict

def train_model(data_loaders, model, loss_func, optimizer, device):
    since = time.time()
    print("training started")
    num_epochs = hparams["num_epochs"]
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    coresp_mcc = 0.0
    coresp_auc = 0.0
    coresp_loss = 0.0
    best_epoch = 0
    stop = False
    lr = optimizer.param_groups[0]["lr"]
    decision = None
    information = EpochInformation(model, device, num_classes, dataset_sizes)
    early_stopper = EarlyStopper(patience=hparams["early_stopping_patience"],
                                 min_delta=hparams["early_stopping_threshold"],
                                 model_weights=copy.deepcopy(model.state_dict()))

    for epoch in range(num_epochs):
        old_time = int(round(time.time() * 1000))
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        if stop == True:
            break
            # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            lr = optimizer.param_groups[0]["lr"]
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode
            information.reset_metrics()
            # Iterate over data.
            #for inputs, labels in data_loaders[phase]:
            data_loader = tqdm(data_loaders[phase])
            for inputs, labels in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    #print shape of tensor
                    #print(outputs.size())
                    #print(labels.size())
                    loss = loss_func(outputs, labels)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                information.update_metrics_for_batch(outputs, loss, inputs, labels)

            result_dict = information.calculate_metrics(phase)
            # prints the all metrics of the training and validation phase
            print(" ".join(name + ": " + str(round(value, 4)) for name, value in result_dict.items()))

            if phase == 'val':
                if early_stopper.early_stop(result_dict["mcc"], copy.deepcopy(model.state_dict())):
                    print('early stopping')
                    stop = True
                if result_dict["acc"] >= best_acc:
                    best_acc = result_dict["acc"]
                    corresp_mcc = result_dict["mcc"]
                    corresp_auc = result_dict["auc"]
                    corresp_loss = result_dict["loss"]
                    best_epoch = epoch
            print("time for " + phase + " of current epoch: " + str(int(round(time.time() * 1000)) - old_time))
        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} in epoch: {best_epoch:4f}')

    # load best model weights
    model.load_state_dict(early_stopper.best_model_weights)
    # model.load_state_dict(best_model_wts)
    return model, best_acc


transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224), antialias=True),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
     #transforms.Normalize(35.37502147246886, 75.87412766890324)])

train_set = GermanCharacterRecognitionDS(path_train_csv, transform=transform, classes=classes,
                                         one_hot_encoder=onehot_encoder, num_channels=3)
test_set = GermanCharacterRecognitionDS(path_test_csv, transform=None, classes=classes,
                                        one_hot_encoder=onehot_encoder, num_channels=3)

#train_loader, val_loader = get_train_and_val_loader(train_set, 250, onehot_encoder, classes)
train_loader, val_loader = split_train_loader(train_set, np.load("train_indices_digits.npy"),
                                              np.load("val_indices_digits.npy"))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print("train_loader: " + str(get_all_labels_of_data_loader(train_loader, onehot_encoder)))
print("val_loader: " + str(get_all_labels_of_data_loader(val_loader, onehot_encoder)))
print("test_loader: " + str(get_all_labels_of_data_loader(test_loader, onehot_encoder)))

data_loaders = {"train": train_loader, "val": val_loader}
dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset)}
#model = network_pytorch.PyTorchClassifier(len(classes))
loss_func = torch.nn.CrossEntropyLoss()


#train_loader = torch.utils.data.DataLoader(
#  torchvision.datasets.MNIST('mnist/', train=True, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#  batch_size=batch_size, shuffle=True)

#test_loader = torch.utils.data.DataLoader(
#  torchvision.datasets.MNIST('mnist/', train=False, download=True,
#                             transform=torchvision.transforms.Compose([
#                               torchvision.transforms.ToTensor(),
#                               torchvision.transforms.Normalize(
#                                 (0.1307,), (0.3081,))
#                             ])),
#    batch_size=batch_size, shuffle=True)
#data_loaders = {"train": train_loader, "val": test_loader}
#dataset_sizes = {"train": len(train_loader.dataset), "val": len(test_loader.dataset)}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.to(device)
optimizer = torch.optim.NAdam(model.parameters(), lr=0.001)
train_model(data_loaders, model, loss_func, optimizer, device)