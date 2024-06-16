"""
Draft of the CNN model for the DNA sequence classification.
"""

import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf

"""
Functions for the onehot encoding
"""

def onehot_encoder(dataset):
    """
    Function that encodes a DNA dataset into a onehot encoding dataset.
    """
    onehot_dataset = [dna_onehot_encoder(dna_string) for dna_string in dataset]
    onehot_dataset_numpy = np.array(onehot_dataset)

    return onehot_dataset_numpy


def dna_onehot_encoder(dna_sequence):
    """
    Function that encodes a single DNA string into a onehot encoding string.
    """
    onehot_dict = {
        'A' : [1, 0, 0, 0],
        'C' : [0, 1, 0, 0],
        'G' : [0, 0, 1, 0],
        'T' : [0, 0, 0, 1]
    }
    encoder = [onehot_dict[nuc] for nuc in dna_sequence]

    return encoder

"""
!- CNN Model
"""

class ConvNet(nn.Module):
    # We can use a differnet pool for each layer
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            # Conv1d for 1D data
            nn.Conv1d(300, 150, kernel_size=3, padding=1),
            nn.ReLU(),
            # Average pool to be more precise or max pool to be more general?
            nn.AvgPool1d(2)
        )
        self.layer2 = nn.Sequential(
            # Conv1d for 1D data
            nn.Conv1d(150, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Average pool to be more precise or max pool to be more general?
            nn.MaxPool1d(2)
        )
        self.linear2 = nn.Linear(64, 2)
    
    def forward(self, x):
        print("Before layer 1: ", x.shape)
        x = self.layer1(x)
        print("After layer 1: ", x.shape)
        x = self.layer2(x)
        print("After layer 2: ", x.shape)
        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)
        print("After flatten: ", x.shape)
        x = self.linear2(x)
        return x

"""
!- MAIN
"""

# Set the device to be used (GPU or CPU)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Device: ", device)

# Read the input from the cvc file
rel_path_train = './data/fullset_train.csv'
rel_path_val = './data/fullset_validation.csv'
rel_path_test = './data/fullset_test.csv'

# Training Set

# Read the input from the csv file
train_csv = pd.read_csv(rel_path_train, sep=",")
# Drop the NaN values
train_csv = train_csv.dropna()
# Describe the data
train_csv.describe()

# Get the data from the csv file
train_data = train_csv.values
# m = number of input samples
m = train_data.shape[0]
print("Amount of data:",m)
X_train = train_data[:m,1]
Y_train = train_data[:m,2].astype(np.int32)

# # Reduce the size of the dataset for testing
m = 90000
X_train = X_train[:m]
Y_train = Y_train[:m]

# OneHot encoding for the training data
print("Start onehot encoding for the training data")
X_train = onehot_encoder(X_train)

# Convert the data to a tensor
X_train = th.from_numpy(X_train)
Y_train = th.tensor(Y_train)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)

# Free memory
del train_csv, train_data, m

# Validation Set

# Read the input from the csv file
val_csv = pd.read_csv(rel_path_val, sep=",")
# Drop the NaN values
val_csv = val_csv.dropna()
# Describe the data
val_csv.describe()

val_data = val_csv.values
# m = number of input samples
m = val_data.shape[0]
print("Amount of data:",m)
X_val = val_data[:m,1]
Y_val = val_data[:m,2].astype(np.int32)

# OneHot encoding for the validation data
print("Start onehot encoding for the validation data")
X_val = onehot_encoder(X_val)

X_val = th.from_numpy(X_val)
Y_val = th.tensor(Y_val)

print("X_val shape", X_val.shape)
print("Y_val shape", Y_val.shape)

# Free memory
del val_csv, val_data, m

# Test

# Read the input from the csv file
test_csv = pd.read_csv(rel_path_test, sep=",")
# Drop the NaN values
test_csv = test_csv.dropna()
# Describe the data
test_csv.describe()

test_data = test_csv.values
# m = number of input samples
m = test_data.shape[0]
print("Amount of data:",m)
X_test = test_data[:m,1]
Y_test = test_data[:m,2].astype(np.int32)

# OneHot encoding for the test data
print("Start onehot encoding for the test data")
X_test = onehot_encoder(X_test)

X_test = th.from_numpy(X_test)
Y_test = th.tensor(Y_test)

print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)

# Free memory
del test_csv, test_data, m

"""
!- TRAIN & VALIDATION
"""

print("Start training the model")

model = ConvNet().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = th.optim.Adam(model.parameters(), lr=0.001)

# Training the model
num_epochs = 7
for epoch in range(num_epochs):
    model.train()
    X_train = X_train.float()
    Y_train = Y_train.long()
    outputs = model(X_train.to(device))
    loss = criterion(outputs, Y_train.to(device))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

print("End training the model")

print("Start validation the model")

# Testing the model
model.eval()
with th.no_grad():
    correct = 0
    total = 0
    X_test = X_test.float()
    Y_test = Y_test.long()
    outputs = model(X_test.to(device))
    _, predicted = th.max(outputs.data, 1)
    total += Y_test.size(0)
    correct += (predicted == Y_test.to(device)).sum().item()

    print(f'Accuracy: {100 * correct / total}%')

print("End validation the model")