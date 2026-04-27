# Generate train, validation, and test datasets
import pandas as pd
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

TEST_SIZE = 0.1
BATCH_SIZE = 256
SEED = 42

user_input = input("Enter Seed: ")
if user_input == "random":
    seed = random.randint(0, 10000)
elif user_input == "SEED":
    seed = SEED
else:
    try:
        seed = int(user_input)
    except ValueError:
        print("Invalid input, defaulting to 42")
        seed = 42
print("New Seed:", seed)
torch.manual_seed(seed)

X_data = pd.read_csv("heartbeats.csv")
y_data = pd.read_csv("labels.csv")

# Dataset
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        data = torch.tensor(self.X.iloc[idx].to_numpy(), dtype=torch.float32).unsqueeze(0)
        label = self.y[idx]
        label = torch.tensor(label, dtype=torch.long)
        return data, label
    
###Stratified Sampling 
labels = y_data.iloc[:, 0]  # assuming labels is a single column

# Encode labels if needed (convert strings like 'N', 'VEB', etc. to integers)
def label_encode(labels):
    label_map = {'N':0, 'VEB':1, 'SVEB':2, 'F':3, 'Q':4}
    labels_encoded = [label_map.get(l) for l in labels]
    return np.array(labels_encoded)
labels_encoded = label_encode(labels)

# Stratified Split
split1 = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_val_idx, test_idx = next(split1.split(X_data, labels_encoded))

# Further split training into training and validation
split_val = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
train_idx, val_idx = next(split_val.split(X_data.iloc[train_val_idx], labels_encoded[train_val_idx]))

# Adjust indices to match the original indices
train_idx = train_val_idx[train_idx]
val_idx = train_val_idx[val_idx]

#Undersampling + SMOTE for training set
# Step 1: undersample N 
under = RandomUnderSampler(sampling_strategy={0: 35000}, random_state=SEED)
# Step 2: oversample abnormal classes (not Q)
smote = SMOTE(
    sampling_strategy={1: 20000, 2: 25000, 3: 20000},  
    random_state=SEED
)
# Step 3: Apply to training set
pipeline = Pipeline(steps=[('under',under),('smote',smote)])
X_train, y_train = X_data.iloc[train_idx], labels_encoded[train_idx]
re_X_train, re_y_train = pipeline.fit_resample(X_train,y_train)

# Datasets
train_dataset = CustomDataset(re_X_train, re_y_train)
val_dataset = CustomDataset(X_data.iloc[val_idx], labels_encoded[val_idx])
test_dataset = CustomDataset(X_data.iloc[test_idx], labels_encoded[test_idx])

print("Datasets Loaded.")