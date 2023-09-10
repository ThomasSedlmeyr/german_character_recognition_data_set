import torchmetrics
import torch

class EarlyStopper:
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

        #update the loss
        self.running_loss += loss.item() * inputs.size(0)
        
    def calculate_metrics(self, phase):
        loss = self.running_loss / self.dataset_sizes[phase]
                
        _, predictions = torch.max(self.running_outputs, 1)
        _, target_indices = torch.max(self.running_labels, 1)
        comparison = predictions == target_indices
        corrects = torch.sum(comparison)
        
        acc = corrects.double() / self.dataset_sizes[phase]
        mcc = self.mcc_metric(predictions, target_indices)
        auc = self.auc_metric(self.running_outputs, target_indices)
        
        # The gradient norm can only be calculated during training
        # Also we calculate the weight-norm only once in each training epoch
        if self.model.training:
            grads = [param.grad.detach().flatten() for param in self.model.parameters() if param.grad is not None]
            l2_norm_grads = torch.linalg.vector_norm(torch.cat(grads))
            weights = [param.detach().flatten() for param in self.model.parameters()]
            l2_norm_weights = torch.linalg.vector_norm(torch.cat(weights)) 
           
        result_dict = {
            "loss" : loss,
            "acc"  : acc.item(),
            "mcc"  : mcc.item(),
            "auc"  : auc.item()
        }
        if self.model.training:
            result_dict["l2_grad"] = l2_norm_grads.item()
            result_dict["l2_weights"] = l2_norm_weights.item()

        return result_dict 