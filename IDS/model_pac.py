import torch.nn as nn

class BaselineCNN(nn.Module):
    def __init__(self, channel_size:int, num_features:int):
        super(BaselineCNN, self).__init__()
        kernel1D_sizes = [3,3,5]
        pooling_kernel_size = 2
        pooling_stride = 2

        # Input layer
        self.c1d1 = nn.Conv1d(in_channels=channel_size, out_channels=64, kernel_size=kernel1D_sizes[0])
        self.c1d_af1 = nn.ReLU()

        self.c1d2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1D_sizes[1])
        self.c1d_af2 = nn.ReLU()

        # Convolutional layer
        self.c1d3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=kernel1D_sizes[2])
        self.c1d_af3 = nn.ReLU()
        self.c1d_pool = nn.MaxPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)

        # Dense layers
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1920, 256)
        self.fc_af1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.fc_af2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 13)

    def forward(self, x):
        # Input layer
        x = self.c1d1(x)
        x = self.c1d_af1(x)

        # Convolutional layer
        x = self.c1d2(x)
        x = self.c1d_af2(x)
        x = self.c1d3(x)
        x = self.c1d_af3(x)
        x = self.c1d_pool(x)

        # Flatten for the dense layers
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        # Dense layers
        x = self.fc1(x)
        x = self.fc_af1(x)
        x = self.fc2(x)
        x = self.fc_af2(x)
        x = self.fc3(x)

        return x
