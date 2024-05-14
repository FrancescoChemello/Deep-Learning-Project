import os
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf
# For the progress bar
from tqdm import tqdm

#!- FUNCTIONS
def onehot_encoder (dataset):
    """
    Function that encodes a DNA dataset into a onehot encoding dataset.
    The dataset is a list of strings, each string is a DNA sequence.
    The function returns a list of lists, each list is a onehot encoding of a DNA sequence.
    """
    # Define the dictionary for the onehot encoding
    onehot_dict = {'A':[1,0,0,0],'C':[0,1,0,0],'G':[0,0,1,0],'T':[0,0,0,1]}
    # Initialize the onehot dataset
    onehot_dataset = np.array([])
    # Iterate over the dataset
    pbar = tqdm(total=len(dataset))
    for sequence in dataset:
        # Initialize the onehot sequence
        onehot_sequence = []
        # Iterate over the sequence
        for base in sequence:
            # Append the onehot base
            onehot_sequence.append(onehot_dict[base])
        # Append the onehot sequence
        onehot_dataset = np.append(onehot_dataset, [onehot_sequence])
        pbar.update(1)
    pbar.close()
    print("Onehot encoding done")
    return onehot_dataset


#!- MAIN

# Set the device to be used (GPU or CPU)
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print("Device: ", device)

# Read the input from the cvc file
absolute_path = os.path.dirname(__file__)
rel_path_train = 'data\\fullset_train.csv'
rel_path_val = 'data\\fullset_validation.csv'
rel_path_test = 'data\\fullset_test.csv'

# Training Set

# Read the input from the csv file
train_csv = pd.read_csv(os.path.join(absolute_path, rel_path_train), sep=",")
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

# OneHot encoding for the training data
print("Start onehot encoding for the training data")
X_train = onehot_encoder(X_train)

# Convert the data to a tensor
X_train = th.from_numpy(np.array(X_train))
Y_train = th.tensor(Y_train)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)

# print("X_train values: ", X_train)
# print("Y_train values: ", Y_train)

# Free memory
del train_csv, train_data, m

exit() # Debug -END OF ONEHOT ENCODING-

# Validation Set

# Read the input from the csv file
val_csv = pd.read_csv(os.path.join(absolute_path, rel_path_val), sep=",")
# Drop the NaN values
val_csv = val_csv.dropna()
# Describe the data
val_csv.describe()

val_data = val_csv.values
# m = number of input samples
m = val_data.shape[0]
print("Amount of data:",m)
X_val = val_data[:m,1]
Y_val = val_data[:m,2]

# OneHot encoding for the validation data
X_val = DNA_to_onehot_dataset(X_val)

X_val = th.from_numpy(np.array(X_val))
Y_val = th.from_numpy(np.array(Y_val))

print("X_val shape", X_val.shape)
print("Y_val shape", Y_val.shape)

print("X_val values: ", X_val)
print("Y_val values: ", Y_val)

# Free memory
del val_csv, val_data, m

# Test

# Read the input from the csv file
test_csv = pd.read_csv(os.path.join(absolute_path, rel_path_test), sep=",")
# Drop the NaN values
test_csv = test_csv.dropna()
# Describe the data
test_csv.describe()

test_data = test_csv.values
# m = number of input samples
m = test_data.shape[0]
print("Amount of data:",m)
X_test = test_data[:m,1]
Y_test = test_data[:m,2]

# OneHot encoding for the test data
X_test = DNA_to_onehot_dataset(X_test)

X_test = th.from_numpy(np.array(X_test))
Y_test = th.from_numpy(np.array(Y_test))

print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)

print("X_test values: ", X_test)
print("Y_test values: ", Y_test)

# Free memory
del test_csv, test_data, m

exit() # Debug -END OF READING DATA-

# # Reshape the data to fit the pytorch model
# train_onehot = th.from_numpy(train_onehot).float()
# val_onehot = th.from_numpy(val_onehot).float()
# test_onehot = th.from_numpy(test_onehot).float()

print("Start training the model")

# RNN

# Hyperparameters
input_size = len(X_train)
hidden_size = 64  # To be defined
output_size = 2  # We want a probabilistic value
num_layers = 1
num_classes = 2
sequence_length = 1
learning_rate = 0.005
batch_size = 8
num_epochs = 10

# 1. Define the model
# Parameters:
#   input_size – The number of expected features in the input x
#   hidden_size – The number of features in the hidden state h
#   num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
#   bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
#   batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
#   dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
#   bidirectional – If True, becomes a bidirectional GRU. Default: 
model = th.nn.GRU(input_size, hidden_size, num_layers, num_classes, num_layers=2, bias=True, batch_first=False, dropout=0.0, bidirectional=True, device=None, dtype=None).to(device)
# 2. Train the model with epochs
criterion = nn.CrossEntropyLoss()   #the most common loss function used for classification problems
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)
# 3. Test the model using epochs

for epoch in range(num_epochs):
    # TODO: Implement the training loop
    break

model.train()

print("End of training the model")

print("Start testing the model")

# 4. Evaluate the model
predicted_values, _ = model(test_data)

print("End of testing the model")
