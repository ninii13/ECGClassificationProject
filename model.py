#Model Definition
import torch
import torch.nn as nn
import torch.nn.functional as F

class SharedKernelConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True):
        super().__init__()
        # Use a single conv layer (1 → out_channels)
        self.base_conv = nn.Conv1d(1, out_channels, kernel_size, stride=stride, bias=bias)
        self.in_channels = in_channels

    def forward(self, x):
        # x: [batch, in_channels, length]
        outs = []
        for i in range(self.in_channels):
            # Process each channel with the same kernel
            xi = x[:, i:i+1, :]              # isolate channel i
            yi = self.base_conv(xi)          # apply shared conv
            outs.append(yi)
        #sum the results
        out = sum(outs)
        return out

class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        # Convolutional Layers
        self.conv1 = SharedKernelConv1d(1, 2, 16, stride=1)
        self.conv2 = SharedKernelConv1d(2, 4, 16, stride=1)
        self.conv3 = SharedKernelConv1d(4, 2, 32, stride=1)

        # Quantization stubs
        self.quant = torch.ao.quantization.QuantStub()
        self.dequant = torch.ao.quantization.DeQuantStub()

        # Dummy input to compute flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            X = self.conv1(dummy_input)
            X = self.conv2(X)
            X = self.conv3(X)
            X = F.max_pool1d(X, 8, stride=8)
            self.flattened_size = X.view(1, -1).shape[1]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 5) 

    def forward(self, X):
        #X = self.quant(X)
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = F.max_pool1d(X, 8, stride=8)
        # Flatten
        X = X.view(X.size(0), -1)
        # Fully Connected Layers
        X = self.fc1(X)
        #X = self.dequant(X)
        return X
