import torch
import copy

from tqdm import tqdm
from torchsummary import summary
from torchvision import transforms

from train_utils import get_train_and_val_set, get_class_counts_of_data_loader
from data_pytorch import GermanCharacterRecognitionDS
from network_pytorch import PyTorchClassifier
from train_utils import EpochInformation
from train_utils import EarlyStopper

"""
This module is for all who also do not like Jupyter Notebooks and only want to run a simple python script for performing
the training. The script is very similar to the Jupyter Notebook training.ipynb. The only difference are some additional 
methods for cleaning up the code.
"""

# Change the paths accordingly
path_train_csv = "/home/thomas/Dokumente/Projekte/german_character_recognition_data_set/train.csv"
path_test_csv = "/home/thomas/Dokumente/Projekte/german_character_recognition_data_set/test.csv"
# First we have to select the classes on which we would like to train on
classes = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
# All available classes which are contained in the dataset
#classes = ['!','$','&','(',')','+','0','1','2','3','4','5','6','7','8','9','<','>','?','A','B','C','D','E','F','G','H',
#           'I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','a','b','c','d','e','f','g','h','i',
#           'j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','~','ß','α','β','π','φ','€','∑','√','∞',
#           '∫']
dict_classes_to_numbers = dict(zip(classes, range(len(classes))))
dict_numbers_to_classes = dict(zip(range(len(classes)), classes))
num_classes = len(classes)
print("Num classes: " + str(num_classes))
num_val_samples_per_class = 250
# Standard DL-parameters
batch_size_train = 128
batch_size_val = 256
num_workers = 2
lr = 0.001
hparams = {"num_epochs": 100, "early_stopping_patience": 5, "early_stopping_threshold": 0.001}
# For getting reproducible results
seed = 0
torch.manual_seed(seed)

def train_model(data_loaders, model, loss_func, optimizer, device):
    print("training started")
    num_epochs = hparams["num_epochs"]
    information = EpochInformation(model, device, num_classes, dataset_sizes)
    early_stopper = EarlyStopper(patience=hparams["early_stopping_patience"],
                                 min_delta=hparams["early_stopping_threshold"],
                                 model_weights=copy.deepcopy(model.state_dict()))
    strop_training = False
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)
        if strop_training == True:
            break
        # Each epoch has a training and validation phase
        for phase in ['val', 'train']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            information.reset_metrics()

            if phase == 'train':
                print("training...")
            else:
                print("validating...")
            data_loader = tqdm(data_loaders[phase])
            for inputs, labels in data_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = loss_func(outputs, labels)
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
                    strop_training = True
    # load best model
    model.load_state_dict(early_stopper.best_model_weights)
    return model


def get_data():
    # We normalize with the men and std of the train set
    standard_transforms = [transforms.ToTensor(), transforms.Normalize(35.37502147246886, 75.87412766890324)]
    test_set = GermanCharacterRecognitionDS(path_test_csv, dict_classes_to_numbers,
                                            transform=transforms.Compose(standard_transforms), classes=classes,
                                            num_channels=1)
    train_set = GermanCharacterRecognitionDS(path_train_csv, dict_classes_to_numbers, transform=None, classes=classes,
                                             num_channels=1)
    # TODO comment the following line after the first run
    train_set, val_set = get_train_and_val_set(train_set, classes, dict_numbers_to_classes)
    # TODO uncomment this line if you want to use the precalculated indnum_val_samples_per_classices which speeds up the run time
    # train_set, val_set = split_train_set_from_indices(train_set, np.load("train_indices.npy"), np.load("val_indices.npy"))
    train_transforms = standard_transforms + [transforms.RandomRotation(30), transforms.RandomGrayscale(p=0.1),
                                              transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))]
    train_set.dataset.transform = transforms.Compose(train_transforms)
    val_set.dataset.transform = transforms.Compose(standard_transforms)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size_train, shuffle=True,
                                               num_workers=num_workers, generator=g)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size_val, shuffle=False, num_workers=num_workers,
                                             generator=g)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size_val, shuffle=False,
                                              num_workers=num_workers,
                                              generator=g)
    class_counts_train = get_class_counts_of_data_loader(train_loader, classes, dict_numbers_to_classes)
    class_counts_val = get_class_counts_of_data_loader(val_loader, classes, dict_numbers_to_classes)
    class_counts_test = get_class_counts_of_data_loader(test_loader, classes, dict_numbers_to_classes)
    print("train_loader: " + str(class_counts_train))
    print("val_loader: " + str(class_counts_val))
    print("test_loader: " + str(class_counts_test))
    data_loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    dataset_sizes = {"train": len(train_loader.dataset), "val": len(val_loader.dataset),
                     "test": len(test_loader.dataset)}
    return train_loader, class_counts_train, data_loaders, dataset_sizes


def evaluate_model(model, device, num_classes, dataset_sizes):
    information_test = EpochInformation(model, device, num_classes, dataset_sizes)
    model.eval()
    for inputs, labels in data_loaders["test"]:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = loss_func(outputs, labels)
        information_test.update_metrics_for_batch(outputs, loss, inputs, labels)
    result_dict = information_test.calculate_metrics("test")
    print("Test metrics:")
    print(" ".join(name + ": " + str(round(value, 4)) for name, value in result_dict.items()))


def calculate_class_weights():
    class_weights = []
    number_train_values = len(train_loader.dataset)
    for class_label in classes:
        weight = float(number_train_values) / class_counts_train[class_label]
        class_weights.append(weight)
    class_weights = torch.tensor(class_weights)
    sum_class_weights = torch.sum(class_weights)
    class_weights = class_weights / sum_class_weights
    print("class weights: ", str(dict(zip(classes, class_weights.tolist()))))
    return class_weights


if __name__ == '__main__':
    # Get the data
    train_loader, class_counts_train, data_loaders, dataset_sizes = get_data()

    # Build the model
    model = PyTorchClassifier(len(classes))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    model.to(device)
    summary(model, (1, 40, 40))

    # Train the model
    class_weights = calculate_class_weights()
    class_weights = class_weights.to(device)
    optimizer = torch.optim.NAdam(model.parameters(), lr=lr)
    loss_func = torch.nn.CrossEntropyLoss(weight=class_weights)
    model = train_model(data_loaders, model, loss_func, optimizer, device)

    # Evaluate the model
    evaluate_model(model, device, num_classes, dataset_sizes)