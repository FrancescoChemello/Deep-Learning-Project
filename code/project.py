import os
import torch as th
import numpy as np
import pandas as pd

# processes an entire dataset of DNA strings to onehot
# -code from VinaMiner-
def DNA_to_onehot_dataset(dataset):
  options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
  onehot_data = []
  for row in dataset:
    onehot_data.append(map(lambda e: options_onehot[e], row))
  onehot_data = np.array(onehot_data)
  print(np.shape(onehot_data))
  return onehot_data 

# processes a DNA string to onehot
# -code from VinaMiner-
def DNA_to_onehot(dna_line):
  options_onehot = {'A': [1,0,0,0,0],'C' :[0,1,0,0,0], 'G':[0,0,1,0,0] ,'T':[0,0,0,1,0],'N':[0,0,0,0,1]}
  onehot_data = map(lambda e: options_onehot[e], dna_line)
  onehot_data = np.array(onehot_data)
  return onehot_data 


#!- MAIN

# Read the input from the cvc file
absolute_path = os.path.dirname(__file__)
rel_path_train = 'data\\fullset_train.csv'
rel_path_val = 'data\\fullset_validation.csv'
rel_path_test = 'data\\fullset_test.csv'
print("I read the data")
# Read the input from the csv file
train_data = pd.read_csv(os.path.join(absolute_path, rel_path_train), delimiter=",",names=["projects","seqs","label","id"])
val_data = pd.read_csv(os.path.join(absolute_path, rel_path_val), delimiter=",",names=["projects","seqs","label","id"])
test_data = pd.read_csv(os.path.join(absolute_path, rel_path_test), delimiter=",",names=["projects","seqs","label","id"])

# Convert the DNA strings to onehot
train_onehot = DNA_to_onehot_dataset(train_data.seqs)
val_onehot = DNA_to_onehot_dataset(val_data.seqs)
test_onehot = DNA_to_onehot_dataset(test_data.seqs)

# Print the first 10 results
for v in range (0, 10):
    print("Train data:"+ train_data.seqs[v])
    print(train_onehot[v])


# 



