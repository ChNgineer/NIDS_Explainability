import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()

        # Input layer
        self.input_layer = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.relu = nn.ReLU()

        # Convolutional layer
        self.conv1d_layer = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3)
        self.pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2)

        # Dense layers
        self.dense_layer1 = nn.Linear(32, 128) # (out_channels of conv1d / kernel_size of pooling, dense size)
        self.relu_dense = nn.ReLU()
        self.dense_layer2 = nn.Linear(128, 12)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # Input layer
        x = self.input_layer(x)
        x = self.relu(x)

        # Convolutional layer
        x = self.conv1d_layer(x)
        x = self.pooling_layer(x)

        # Flatten for the dense layers
        x = x.view(x.size(0), -1)

        # Dense layers
        x = self.dense_layer1(x)
        x = self.relu_dense(x)
        x = self.dense_layer2(x)
        x = self.softmax(x)

        return x
