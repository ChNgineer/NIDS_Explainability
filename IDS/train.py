from pathlib import *
import json

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from collections import Counter

from data.dataset import CICDDoS2019Dataset, min_max_normalize
from model import BaselineCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def get_accuracy(predictions, labels):
    predictions = torch.argmax(predictions,dim=1)
    if predictions.shape != labels.shape:
        raise ValueError(f"Predictions and labels must have the same shape. predictions of size {predictions.shape} and labels of size {labels.shape}")
    
    correct = (predictions == labels).sum().item()

    total = labels.numel()
    accuracy = correct / total

    return accuracy

def run_training(dataset:Dataset, epochs:int, split:list[float]=[0.8,0.2], batch_size:int=64, lr:int=0.001, checkpoint:int=None, random_seed=None):
    training_stats = {'train_pred': [], 'train_labels': [], 'train_loss': [], 'val_pred': [], 'val_labels': [], 'val_loss': []}

    if seed:
        np.random.seed(seed)
    if len(split) == 2:
        train_set, val_set = random_split(dataset, split)
        test_set = None
    elif len(split) == 3:
        train_set, val_set, test_set = random_split(dataset, split)
        training_stats.update({'test_pred': [], 'test_labels': [], 'test_loss': []})
    else:
        ValueError("split parameter in training must be of length 2 or 3")

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=8) 
    if test_set:
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, num_workers=8) 

    model = BaselineCNN(channel_size=dataset.data.shape[-2], num_features=dataset.data.shape[-1])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    for epoch in tqdm(range(epochs), desc='epochs'):
        model.train()

        train_predictions = []
        train_targets = []

        total_loss = 0.0
        total_accuracy = 0.0

        for inputs, labels in tqdm(train_loader, desc='batches', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_accuracy += get_accuracy(outputs.detach().cpu(), labels.detach().cpu())
            train_predictions.extend(torch.argmax(outputs.detach().cpu(),dim=1).numpy().tolist())
            train_targets.extend(labels.detach().cpu().numpy().tolist())
        avg_loss = total_loss / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        training_stats['train_pred'].append(train_predictions)
        training_stats['train_labels'].append(train_targets)
        training_stats['train_loss'].append(avg_loss)

        print(f'\nEpoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {avg_accuracy:.4f}')

        model.eval()
        with torch.no_grad():
            val_predictions = []
            val_targets = []
            total_val_loss = 0.0
            total_val_accuracy = 0.0
            for val_data, val_labels in val_loader:
                val_data, val_labels = val_data.to(device), val_labels.to(device)
                val_outputs = model(val_data)
                val_loss = criterion(val_outputs, val_labels)
                val_predictions.extend(torch.argmax(val_outputs.detach().cpu(),dim=1).numpy().tolist())
                val_targets.extend(val_labels.cpu().numpy().tolist())
                total_val_loss += val_loss.item()
                total_val_accuracy += get_accuracy(val_outputs.detach().cpu(), val_labels.detach().cpu())
            val_loss = total_val_loss / len(val_loader)
            val_accuracy = total_val_accuracy / len(val_loader)
            training_stats['val_pred'].append(val_predictions)
            training_stats['val_labels'].append(val_targets)
            training_stats['val_loss'].append(val_loss)
            
            print(f'Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')

            if checkpoint:
                if (epoch + 1) % checkpoint == 0:
                    with open(f'training_metrics/packet_stats_lr1e-2/training_stats_epc{epoch+1}.json', 'w') as json_file:
                        json.dump(training_stats, json_file)
                    torch.save(model.state_dict(), f'saved_models/packet_models_lr1e-2/baseline_epc{epoch+1}.pt')

    if test_set:
        model.eval()
        with torch.no_grad():
            test_predictions = []
            test_targets = []
            total_test_loss = 0.0
            total_test_accuracy = 0.0
            for test_data, test_labels in test_loader:
                test_data, test_labels = test_data.to(device), test_labels.to(device)
                test_outputs = model(test_data)
                test_loss = criterion(test_outputs, test_labels)
                test_predictions.extend(torch.argmax(val_outputs.detach().cpu(),dim=1).numpy().tolist())
                test_targets.extend(test_labels.cpu().numpy().tolist())
                total_test_loss += test_loss.item()
                total_test_accuracy += get_accuracy(test_outputs.detach().cpu(), test_labels.detach().cpu())
            test_loss = total_test_loss / len(test_loader)
            test_accuracy = total_test_accuracy / len(test_loader)
            training_stats['test_pred'] = test_predictions
            training_stats['test_labels'] = test_targets
            training_stats['test_loss'] = test_loss
            
        print(f'Epoch [{epoch + 1}/{epochs}], Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

    torch.save(model.state_dict(), 'saved_models/packet_models_lr1e-2/baseline_final.pt')
    with open('training_metrics/packet_stats_lr1e-2/training_stats_final.json', 'w') as json_file:
        json.dump(training_stats, json_file)
    

if __name__ == '__main__':
    seed = 774

    # Hyperparameters
    batch_size = 256
    learning_rate = 0.01 # Remember to maybe change this back to 0.0001
    epochs = 500

    with open('data/utils/filters.json','r') as f:
        filters = json.load(f)
        
    data_dir = "data/Processed Data/"
    ddos_dataset = CICDDoS2019Dataset(data_dir, transform=F.normalize, transform_args=[2.0,0], balanced=True, random_seed=seed) 
    run_training(ddos_dataset, epochs, batch_size=batch_size, lr=learning_rate, checkpoint=50, random_seed=seed)