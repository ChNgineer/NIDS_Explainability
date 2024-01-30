import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from data import CICDDoS2019Dataset  # Import your custom dataset class

# Assuming you have a custom 1D CNN model defined as BaselineCNN
from model import BaselineCNN

# Hyperparameters
batch_size = 32
learning_rate = 0.001
epochs = 10

train_dataset = CICDDoS2019Dataset("*.csv")
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

model = BaselineCNN()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
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

torch.save(model.state_dict(), 'your_model.pth')
