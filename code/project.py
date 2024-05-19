import os
import threading
import time
import sys

import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf
# For the progress bar
from tqdm import tqdm

# Global variable to stop the progress bar (. -> .. -> ... -> .)
global stop_progress
stop_progress = False

#!- FUNCTIONS
def show_progress():
    global stop_progress
    i = 0
    while not stop_progress:
        if i < 3:
            sys.stdout.write('\r')
            sys.stdout.write("Operation in progress" + ("." * (i+1)))
            sys.stdout.flush()
            i += 1
        else:
            sys.stdout.write('\r')
            sys.stdout.flush()
            i = 0
        time.sleep(0.2)  # Wait

def onehot_encoder (dataset):
    """
    Function that encodes a DNA dataset into a onehot encoding dataset.
    The dataset is a list of strings, each string is a DNA sequence.
    The function returns a list of lists, each list is a onehot encoding of a DNA sequence.
    """
    # Define the dictionary for the onehot encoding
    onehot_dict = {'A':[1,0,0,0], 'C':[0,1,0,0], 'G':[0,0,1,0], 'T':[0,0,0,1]}
    # Define the chunk size for the export of the data to a numpy array
    chunk_size = 10000
    # Initialize the onehot dataset
    onehot_dataset = []
    # Iterate over the dataset
    pbar = tqdm(total=len(dataset))
    for sequence in dataset:
        # Initialize the onehot sequence
        onehot_sequence = np.array([])
        # Iterate over the sequence
        for base in sequence:
            # Append the onehot base
            onehot_sequence = np.append(onehot_sequence, onehot_dict[base])
        # Append the onehot sequence
        onehot_dataset.append(onehot_sequence)
        pbar.update(1)
    # Convert the onehot dataset to a numpy array to have an arry of arrays
    pbar.close()
    print("Starting convertion to numpy array")
    sys.stdout.flush()
    global stop_progress
    def concatenate_chunks(chunks):
        return np.concatenate(chunks)

    t = threading.Thread(target=show_progress)
    t.start()
    # Convert the onehot dataset to a numpy array in chunks to avoid memory issues
    onehot_dataset_numpy = np.empty([0, 1200])
    for i in range(0, len(onehot_dataset), chunk_size):
        if i+chunk_size < len(onehot_dataset):
            chunk = concatenate_chunks((onehot_dataset_numpy, np.asarray(onehot_dataset[i:i+chunk_size])))
        else:
           chunk = concatenate_chunks((onehot_dataset_numpy, np.asarray(onehot_dataset[i:])))
        onehot_dataset_numpy = np.concatenate((onehot_dataset_numpy, chunk), axis=0)
    # onehot_dataset = np.asarray(onehot_dataset)
    stop_progress = True
    t.join()
    print("\r   \r", end="")
    print("\nType of onehot_dataset_numpy: ", type(onehot_dataset_numpy))
    print("Shape of onehot_dataset_numpy: ", onehot_dataset_numpy.shape)
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

# Reduce the size of the dataset for testing
m = 100000 
X_train = X_train[:m]
Y_train = Y_train[:m]

# OneHot encoding for the training data
print("Start onehot encoding for the training data")
X_train = onehot_encoder(X_train)

# Convert the data to a tensor
X_train = th.tensor(X_train)
Y_train = th.tensor(Y_train)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)

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
X_val = onehot_encoder(X_val)

X_val = th.from_numpy(np.array(X_val))
Y_val = th.from_numpy(np.array(Y_val))

print("X_val shape", X_val.shape)
print("Y_val shape", Y_val.shape)

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
X_test = onehot_encoder(X_test)

X_test = th.from_numpy(np.array(X_test))
Y_test = th.from_numpy(np.array(Y_test))

print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)

# Free memory
del test_csv, test_data, m

exit() # Debug -END OF READING DATA-S

print("Start training the model")

# RNN

# 1. Define the model
# Hyperparameters for the model (need a vector in a k-fold validation):
#   input_size – The number of expected features in the input x
#   hidden_size – The number of features in the hidden state h
#   num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
#   bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
#   batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
#   dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
#   bidirectional – If True, becomes a bidirectional GRU. Default: 

input_size = 4  # A, C, G, T
hidden_size = 64  # To be defined
output_size = 2  # We want a probabilistic value
num_layers = 1  # Just one GRU and not n GRU stacked
bias = True # We want to use bias
batch_first = False # The input and output tensors are provided as (seq, batch, feature)
dropout = 0.0 # No dropout
bidirectional = True # We want a bidirectional GRU

# INPUT: input, h_0
# -input: tensor shape (L, H_in) where L is the sequence length and H_in is the input size.
# -h_0: tensor shape (D*num_layers, H_out) where D is 2 for bidirectional and H_out is the hidden size.

# OUTPUT: output, h_n
# -output: tensor shape (L, D*H_out) where L is the sequence length, D is 2 for bidirectional and H_out is the hidden size.
# -h_n: tensor shape (D*num_layers, H_out) where D is 2 for bidirectional and H_out is the hidden size.

learning_rate = 0.001
num_classes = 2

model = th.nn.GRU(input_size, hidden_size, num_layers, num_classes, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional, device=None, dtype=None).to(device)
criterion = nn.CrossEntropyLoss()   #the most common loss function used for classification problems
optimizer = th.optim.Adam(model.parameters(), lr=learning_rate)

# 2. Train the model with epochs
# for epoch in range(num_epochs):
#     # TODO: Implement the training loop
#     break

model.train()
# 3. Test the model using epochs

print("End of training the model")

print("Start testing the model")

# 4. Evaluate the model
predicted_values, _ = model(test_data)

print("End of testing the model")
