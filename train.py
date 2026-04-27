
"""Train and Test Data"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
import torch.nn as nn
import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import random

from model import ConvolutionalNetwork
from datasets import SEED, test_dataset, train_dataset, val_dataset
from evaluate import evaluate

X_data = pd.read_csv("heartbeats.csv")
y_data = pd.read_csv("labels.csv")

TEST_SIZE = 0.1
BATCH_SIZE = 256
torch.manual_seed(SEED)

###
# Load Dataset
# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

###
input_length = X_data.shape[1]
model = ConvolutionalNetwork(input_length)
# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#Loss Function Optimizier
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

import time
start_time = time.time()
###
# Create Variables to Track Things
epochs =  10
train_losses = []
test_losses = []
train_correct = []
test_correct = []

# For Loop of Epochs
for i in range(epochs):
  trn_corr = 0
  tst_corr = 0

  #Train
  for b, (X_train, y_train) in enumerate(train_loader):
    b+=1 #start batches at 1

    # Move batch to GPU if available
    X_train, y_train = X_train.to(device), y_train.to(device)

    # Forward pass
    y_pred = model(X_train)
    loss = criterion(y_pred, y_train)

    predicted = torch.max(y_pred.data, 1)[1] #1 inside torch.max means 1 max per batch, 1 inside [] means the index of the max value ([0] is the max value)
    batch_corr = (predicted == y_train).sum()
    trn_corr += batch_corr

    #Update our Optiimizers
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Print out results
    if b%600 == 0:
      print(f'Epoch: {i}  Batch: {b}  Loss: {loss.item()}')

  train_losses.append(loss)
  train_correct.append(trn_corr)

  #Test
  with torch.no_grad(): #no gradients so weights and biases won't update with test data
    for b, (X_test, y_test) in enumerate(val_loader):
      # Move batch to GPU if available
      X_test, y_test = X_test.to(device), y_test.to(device)

      y_val = model(X_test)
      predicted = torch.max(y_val.data, 1)[1] #Adding up correct predictions
      tst_corr += (predicted == y_test).sum()

  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_correct.append(tst_corr)
  print(f'Epoch: {i}  Loss: {loss.item()}')

###
current_time = time.time()
elapsed_time = current_time - start_time
print(f'Training took: {elapsed_time/60} minutes!')
print("=================== Original Model Evaluation ===================")
model.eval()
evaluate(model, test_loader)

# Save the model weights
model.eval()
torch.save(model.state_dict(), f"{SEED}_norelu.pth")
model.eval()
weights = model.state_dict()

