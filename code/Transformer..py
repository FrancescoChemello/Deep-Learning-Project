import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
import os
import pandas as pd
import numpy as np

class ViraminerDataset(Dataset):
    def __init__ (self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__ (self, idx):
        return self.sequences[idx], self.labels[idx]
    
class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, output_dim):
        super(TransformerModel, self).__init__()
        self.model_dim = model_dim
        self.embedding = nn.Embedding(vocab_size, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim)
        self.transformer = nn.Transformer(d_model=model_dim,
                                          nhead=num_heads,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src = self.embedding(src) * math.sqrt(self.model_dim)
        tgt = self.embedding(tgt) * math.sqrt(self.model_dim)
        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)
        transformer_out = self.transformer(src, tgt, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        out = self.fc_out(transformer_out)
        return out

#!Main
# get the file using the absolute path
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
rel_path_train = os.path.join(data_dir, 'fullset_train.csv')
rel_path_val = os.path.join(data_dir, 'fullset_validation.csv')
rel_path_test = os.path.join(data_dir, 'fullset_test.csv')

# read the data
train_csv = pd.read_csv(rel_path_train, sep=',')
val_csv = pd.read_csv(rel_path_val, sep=',')
test_csv = pd.read_csv(rel_path_test, sep=',')

#drop the NaN values
train_csv = train_csv.dropna()
val_csv = val_csv.dropna()
test_csv = test_csv.dropna()

# get the sequences and labels
train_data = train_csv.values
val_data = val_csv.values
test_data = test_csv.values

#get the sequences and labels
m = train_data.shape[0]
train_sequences = train_data[:m, 1]
train_labels = train_data[:m, 2]
train_sequences = np.array([x for x in train_sequences], dtype=np.string_)
train_labels = np.array([int(x) for x in train_labels], dtype=np.int32)
train_sequences = torch.from_numpy(train_sequences)
train_labels = torch.tensor(train_labels, dtype=torch.long)

m = val_data.shape[0]
val_sequences = val_data[:m, 1]
val_labels = val_data[:m, 2]
val_sequences = torch.tensor(val_sequences, dtype=torch.long)
val_labels = torch.tensor(val_labels, dtype=torch.long)

m = test_data.shape[0]
test_sequences = test_data[:m, 1]
test_labels = test_data[:m, 2]
test_sequences = torch.tensor(test_sequences, dtype=torch.long)
test_labels = torch.tensor(test_labels, dtype=torch.long)

#free the memory
del train_data, val_data, test_data, train_csv, val_csv, test_csv, m, rel_path_train, rel_path_val, rel_path_test, data_dir

# create the dataset
train_dataset = ViraminerDataset(train_sequences, train_labels)
val_dataset = ViraminerDataset(val_sequences, val_labels)
test_dataset = ViraminerDataset(test_sequences, test_labels)

#free the memory
del train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels

# create the dataloader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

#free the memory
del train_dataset, val_dataset, test_dataset

# initialize the parameters
vocab_size = 4
model_dim = 512
num_heads = 8
num_encoder_layers = 6
num_decoder_layers = 6
dim_feedforward = 2048
dropout = 0.1
output_dim = 2
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# create the model
model = TransformerModel(vocab_size, model_dim, num_heads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout, output_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

# train the model
num_epochs = 1

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    
    for src, tgt in train_loader:
        # src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        
        # Create masks
        src_mask = (src != 0).unsqueeze(-2).to(device) 
        tgt_input = src[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = model(src, tgt_input, src_mask, src_mask)
        loss = criterion(output.view(-1, output_dim), tgt_output.view(-1))
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Train Loss: {epoch_loss / len(train_loader)}')
    
    # Validation loop
    model.eval()
    val_loss = 0
    
    with torch.no_grad():
        for src, tgt in val_loader:
            src, tgt = src.to(device), tgt.to(device)
            
            # Create masks
            src_mask = (src != 0).unsqueeze(-2).to(device)
            tgt_input = src[:, :-1]
            tgt_output = tgt[:, 1:]
            
            output = model(src, tgt_input, src_mask, src_mask)
            loss = criterion(output.view(-1, output_dim), tgt_output.view(-1))
            
            val_loss += loss.item()
    
    print(f'Epoch {epoch+1}, Val Loss: {val_loss / len(val_loader)}')

# Testing loop
model.eval()
test_loss = 0

with torch.no_grad():
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Create masks
        src_mask = (src != 0).unsqueeze(-2).to(device)
        tgt_input = src[:, :-1]
        tgt_output = tgt[:, 1:]
        
        output = model(src, tgt_input, src_mask, src_mask)
        loss = criterion(output.view(-1, output_dim), tgt_output.view(-1))
        
        test_loss += loss.item()

print(f'Test Loss: {test_loss / len(test_loader)}')