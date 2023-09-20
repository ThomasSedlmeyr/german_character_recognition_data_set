import numpy as np
import torchmetrics
import torch

from torch.utils.data import Subset

class EarlyStopper:
    """
    Custom early stopper
    """
    def __init__(self, patience=5, min_delta=0.001, model_weights=None):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_validation_acc = 0
        self.best_model_weights = model_weights

    def early_stop(self, validation_acc, model_weights):
        # If the model improved we store the best weights
        if validation_acc > self.best_validation_acc:
            self.best_model_weights = model_weights

        # If the model improved more than the threshold
        if validation_acc > self.best_validation_acc + self.min_delta:
            self.counter = 0
            self.best_validation_acc = validation_acc
        else:
            if self.counter >= self.patience:
                return True
            self.counter += 1
        return False


class EpochInformation:
    """
    For tracking the metrics during training and validation
    """
    def __init__(self, model, device, num_classes, dataset_sizes):
        self.mcc_metric = torchmetrics.MatthewsCorrCoef(task='multiclass', num_classes=num_classes).to(device)
        self.auc_metric = torchmetrics.AUROC(task='multiclass', num_classes=num_classes).to(device)
        self.dataset_sizes = dataset_sizes
        self.running_loss = 0.0
        self.running_outputs = None
        self.running_labels = None
        self.model = model

    def reset_metrics(self):
        self.running_loss = 0.0
        self.running_outputs = None
        self.running_labels = None

    def update_metrics_for_batch(self, outputs, loss, inputs, labels):
        if self.running_outputs is None:
            self.running_outputs = outputs
        else:
            self.running_outputs = torch.cat((self.running_outputs, outputs), dim=0)

        if self.running_labels is None:
            self.running_labels = labels
        else:
            self.running_labels = torch.cat((self.running_labels, labels), dim=0)

        # update the loss
        self.running_loss += loss.item() * inputs.size(0)

    def calculate_metrics(self, phase):
        loss = self.running_loss / self.dataset_sizes[phase]

        _, predictions = torch.max(self.running_outputs, 1)
        comparison = predictions == self.running_labels
        corrects = torch.sum(comparison)

        acc = corrects.double() / self.dataset_sizes[phase]
        mcc = self.mcc_metric(predictions, self.running_labels)
        auc = self.auc_metric(self.running_outputs, self.running_labels)

        # The gradient norm can only be calculated during training
        # Also we calculate the weight-norm only once in each training epoch
        if self.model.training:
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            l2_norm_grads = torch.linalg.vector_norm(torch.cat(grads))
            weights = [param.detach().flatten() for param in self.model.parameters()]
            l2_norm_weights = torch.linalg.vector_norm(torch.cat(weights))

        result_dict = {
            "loss": loss,
            "acc": acc.item(),
            "mcc": mcc.item(),
            "auc": auc.item()
        }
        if self.model.training:
            result_dict["l2_grad"] = l2_norm_grads.item()
            result_dict["l2_weights"] = l2_norm_weights.item()

        return result_dict


def get_train_and_val_set(train_data_set, classes, dict_numbers_to_classes, num_samples_validation_data=250):
    """
    Splits the train_data_set into a train- and a validation-set
    :param train_data_set: the dataset which should be split
    :param classes: The class labels in the dataset
    :param dict_numbers_to_classes:  A dictionary which maps the class labels to numbers
    :param num_samples_validation_data: The number of samples per class which should be contained in the validation set
    :return: The train- and validation-set
    """
    print("Splitting train- and val-data ...")
    val_count = dict(zip(classes, len(classes) * [0]))
    val_indices = []
    train_indices = []
    for i in range(len(train_data_set)):
        _, label = train_data_set[i]
        label_string = dict_numbers_to_classes[label]
        if val_count[label_string] < num_samples_validation_data:
            val_count[label_string] += 1
            val_indices.append(i)
        else:
            train_indices.append(i)

    np.save("val_indices.npy", np.asarray(val_indices))
    np.save("train_indices.npy", np.asarray(train_indices))
    train_set, val_set = split_train_set_from_indices(train_data_set, train_indices, val_indices)
    print("Splitting done")
    return train_set, val_set

def split_train_set_from_indices(train_data_set, train_indices, val_indices):
    train_set = Subset(train_data_set, train_indices)
    val_set = Subset(train_data_set, val_indices)
    return train_set, val_set

def get_class_counts_of_data_loader(data_loader, classes, dict_numbers_to_classes):
    labels_count_dict = dict(zip(classes, len(classes) * [0]))
    for _, labels in data_loader:
        string_labels = [dict_numbers_to_classes[number] for number in labels.tolist()]
        for label in string_labels:
            labels_count_dict[label] += 1
    return labels_count_dict

