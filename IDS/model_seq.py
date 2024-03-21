import torch.nn as nn

class SequenceFeatureCNN(nn.Module):
    def __init__(self, channel_size:int, feature_size:int):
        super(SequenceFeatureCNN, self).__init__()
        seq_length = feature_size
        kernel1D_sizes = [5,5]
        pooling_kernel_size = 2
        pooling_stride = 2

        # Input layer
        self.input_layer = nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel1D_sizes[0])
        self.relu = nn.ReLU()

        # Convolutional layer
        self.conv1d_layer = nn.Conv1d(in_channels=channel_size, out_channels=channel_size, kernel_size=kernel1D_sizes[1])
        self.pooling_layer = nn.AvgPool1d(kernel_size=pooling_kernel_size, stride=pooling_stride)

        # Dense layers
        dense_input_size = int(channel_size * (seq_length - sum(kernel1D_sizes) + len(kernel1D_sizes)) / pooling_kernel_size)
        self.dense_layer1 = nn.Linear(dense_input_size, 128)
        self.relu_dense = nn.ReLU()
        self.dense_layer2 = nn.Linear(128, 13)

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

        return x
