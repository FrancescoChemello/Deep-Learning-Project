import os
import torch as th
import torch.nn as nn
import numpy as np
import pandas as pd
import tensorflow as tf

# processes an entire dataset of DNA strings to onehot
# # -code from VinaMiner-
# def DNA_to_onehot_dataset(dataset):
#   options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
#   onehot_data = []
#   for row in dataset:
#     onehot_data.append(map(lambda e: options_onehot[e], row))
#   onehot_data = np.array(onehot_data)
#   print(np.shape(onehot_data))
#   return onehot_data 

# # processes a DNA string to onehot
# # -code from VinaMiner-
# def DNA_to_onehot(dna_line):
#   options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
#   onehot_data = map(lambda e: options_onehot[e], dna_line)
#   onehot_data = np.array(onehot_data)
#   return onehot_data 


#!- MAIN

# Read the input from the cvc file
absolute_path = os.path.dirname(__file__)
rel_path_train = 'data\\fullset_train.csv'
rel_path_val = 'data\\fullset_validation.csv'
rel_path_test = 'data\\fullset_test.csv'
print("I read the data")
# Read the input from the csv file
# DNA string
train_csv = pd.read_csv(os.path.join(absolute_path, rel_path_train), sep=",")
val_csv = pd.read_csv(os.path.join(absolute_path, rel_path_val), sep=",")
test_csv = pd.read_csv(os.path.join(absolute_path, rel_path_test), sep=",")

# Drop the NaN values
train_csv = train_csv.dropna()
val_csv = val_csv.dropna()
test_csv = test_csv.dropna()

# Describe the data
train_csv.describe()
val_csv.describe()
test_csv.describe()

# Training

train_data = train_csv.values
# m = number of input samples
m = train_data.shape[0]
print("Amount of data:",m)
X_train = train_data[:m,1]
Y_train = train_data[:m,2]

X_train = np.array(X_train)
Y_train = np.array(Y_train)

print("X_train shape: ", X_train.shape)
print("Y_train shape: ", Y_train.shape)

print("X_train values: ", X_train)
print("Y_train values: ", Y_train)

# Validation

val_data = val_csv.values
# m = number of input samples
m = val_data.shape[0]
print("Amount of data:",m)
X_val = val_data[:m,1]
Y_val = val_data[:m,2]

X_val = np.array(X_val)
Y_val = np.array(Y_val)

print("X_val shape", X_val.shape)
print("Y_val shape", Y_val.shape)

print("X_val values: ", X_val)
print("Y_val values: ", Y_val)

# Test

test_data = test_csv.values
# m = number of input samples
m = test_data.shape[0]
print("Amount of data:",m)
X_test = test_data[:m,1]
Y_test = test_data[:m,2]

X_test = np.array(X_test)
Y_test = np.array(Y_test)

print("X_test shape", X_test.shape)
print("Y_test shape", Y_test.shape)

print("X_test values: ", X_test)
print("Y_test values: ", Y_test)

exit() # Debug -END OF READING DATA-

# # Reshape the data to fit the pytorch model
# train_onehot = th.from_numpy(train_onehot).float()
# val_onehot = th.from_numpy(val_onehot).float()
# test_onehot = th.from_numpy(test_onehot).float()

# RNN

input_size = len(train_data)
hidden_size = 64  # To be defined
output_size = 1  # We want a probabilistic value

# 1. Define the model
# Parameters:
#   input_size – The number of expected features in the input x
#   hidden_size – The number of features in the hidden state h
#   num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two GRUs together to form a stacked GRU, with the second GRU taking in outputs of the first GRU and computing the final results. Default: 1
#   bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
#   batch_first – If True, then the input and output tensors are provided as (batch, seq, feature) instead of (seq, batch, feature). Note that this does not apply to hidden or cell states. See the Inputs/Outputs sections below for details. Default: False
#   dropout – If non-zero, introduces a Dropout layer on the outputs of each GRU layer except the last layer, with dropout probability equal to dropout. Default: 0
#   bidirectional – If True, becomes a bidirectional GRU. Default: False
GRU_model = th.nn.GRU(input_size, hidden_size, num_layers=2, bias=True, batch_first=False, dropout=0.0, bidirectional=True, device=None, dtype=None)
# 2. Train the model with epochs
criterion = nn.MSELoss()
optimizer = th.optim.Adam(GRU_model.parameters(), lr=0.001)
# 3. Test the model using epochs

for epoch in range(10):
    for inputs in train_data:  # Carica i dati di addestramento
        # Print the first item of train_data to see its structure
        print(next(iter(train_data)))
        optimizer.zero_grad()
        outputs, _ = GRU_model(inputs)
        loss = criterion(outputs[:, -1, :], targets)
        loss.backward()
        optimizer.step()

# 4. Evaluate the model
predicted_values, _ = GRU_model(test_data)


