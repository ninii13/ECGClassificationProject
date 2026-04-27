# plot_confusion_matrix.py

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Dataset definition
# ----------------------------
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = torch.tensor(self.X.iloc[idx].to_numpy(), dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.y[idx], dtype=torch.long)
        return data, label

# ----------------------------
# Model definition (same as training)
# ----------------------------
class SharedKernelConv1d(nn.Module):
    """
    A quantization-friendly 1D convolution layer that shares the same kernel
    across all input channels for each output channel.
    Internally uses nn.Conv1d to stay FX-compatible.
    """
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
        # Average or sum the results (use average to keep scale stable)
        out = sum(outs)
        return out
    
class ConvolutionalNetwork(nn.Module):
    def __init__(self, input_length):
        super().__init__()
        # Convolutional Layers
        self.conv1 = SharedKernelConv1d(1, 2, 16, stride=1)
        self.conv2 = SharedKernelConv1d(2, 4, 16, stride=1)
        self.conv3 = SharedKernelConv1d(4, 2, 32, stride=1)

        # Dummy input to compute flattened feature size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_length)
            x = F.relu(self.conv1(dummy_input))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = F.max_pool1d(x, 8, stride=8)
            self.flattened_size = x.view(1, -1).shape[1]

        # Fully Connected Layers
        self.fc1 = nn.Linear(self.flattened_size, 5) 

    def forward(self, X):
        X = self.conv1(X)
        X = self.conv2(X)
        X = self.conv3(X)
        X = F.max_pool1d(X, 8, stride=8)
        # Flatten
        X = X.view(X.size(0), -1)

        # Fully Connected Layers
        X = self.fc1(X)

        return X

# ----------------------------
# Load data
# ----------------------------
X_data = pd.read_csv("heartbeats.csv")
y_data = pd.read_csv("labels.csv")
labels = y_data.iloc[:, 0]
label_map = {'N':0, 'VEB':1, 'SVEB':2, 'F':3, 'Q':4}
def label_encode(labels):
    labels_encoded = [label_map.get(l) for l in labels]
    return np.array(labels_encoded)
labels_encoded = label_encode(labels)

# ----------------------------
# Recreate test split
# ----------------------------
TEST_SIZE = 0.1
SEED = 42
split1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
_, test_idx = next(split1.split(X_data, labels_encoded))

test_dataset = CustomDataset(X_data.iloc[test_idx], labels_encoded[test_idx])
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# ----------------------------
# Load model
# ----------------------------
input_length = X_data.shape[1]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvolutionalNetwork(input_length)
model.load_state_dict(torch.load("42_norelu.pth", map_location=device))
model.to(device)
model.eval()

# ----------------------------
# Compute predictions
# ----------------------------
all_preds = []
all_labels = []

with torch.no_grad():
    for X_test, y_test in test_loader:
        X_test, y_test = X_test.to(device), y_test.to(device)
        outputs = model(X_test)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_test.cpu().numpy())

# ----------------------------
# Confusion matrix
# ----------------------------
cm = confusion_matrix(all_labels, all_preds)
class_names = label_map.keys()

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# ----------------------------
# Confusion matrix (normalized)
# ----------------------------
cm = confusion_matrix(all_labels, all_preds, normalize='true')  # normalize by row
class_names = label_map.keys()  # ['F', 'N', 'Q', 'SVEB', 'VEB']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45, values_format=".2f")
plt.title("Normalized Confusion Matrix (per class)")
plt.show()