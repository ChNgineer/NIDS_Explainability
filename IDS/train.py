import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from data import CICDDoS2019Dataset
from pathlib import *

from model import BaselineCNN

seed = 42

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

dir_0112 = Path("DDoS Evaluation Dataset (CIC-DDoS2019)/01-12/")
dir_0311 = Path("DDoS Evaluation Dataset (CIC-DDoS2019)/03-11/")
dataset = CICDDoS2019Dataset([dir_0112,dir_0311])

generator = torch.Generator().manual_seed(seed)
lengths = [int(len(dataset)*0.6), int(len(dataset)*0.2), int(len(dataset)*0.2)]
lengths[2] += len(dataset) - sum(lengths)
train_set, val_set, test_set = random_split(dataset, lengths, generator)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)

model = BaselineCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in tqdm(range(epochs)):
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_loader)
    print(f'Epoch [{epoch + 1}/{epochs}], Loss: {average_loss:.4f}')

torch.save(model.state_dict(), 'baseline.pth')
