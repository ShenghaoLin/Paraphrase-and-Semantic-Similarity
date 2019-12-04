import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math

from model import *

import random
import sys
import pickle
import argparse

device = torch.device(0 if torch.cuda.is_available() else "cpu")

d_model=50
DATA_PATH = '../data/train_data_'+str(d_model)+'d.pkl'
f=open(DATA_PATH, "rb")
x0=pickle.load(f)
x1=pickle.load(f)
Y=pickle.load(f)
f.close()
d_len=x0.size()[0]
s_len=x0.size()[1]
    
def make_model(d_model, max_len, N=6, d_ff=256, h=5, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = AttentionEnc(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        c(position), d_model, max_len)
        
    
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def generate_batch(x0, x1, Y, batch_size, start_i):
    l=min(batch_size, len(x0)-start_i)
    x0_batch=[]
    x1_batch=[]
    Y_batch=[]
    for i in range(l):
        x0_batch.append(x0[start_i+i].numpy())
        x1_batch.append(x1[start_i+i].numpy())
        Y_batch.append(Y[start_i+i])
    x0_batch=torch.Tensor(x0_batch).float().to(device)
    x1_batch=torch.tensor(x1_batch).float().to(device)
    Y_batch=torch.tensor(Y_batch).long().to(device)
    return x0_batch, x1_batch, Y_batch

model = make_model(d_model, s_len).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate=1e-3
reg=1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
tot_epoch=5000
batch_size=100
batch_arrange=[i for i in range(0, d_len, batch_size)]
# batch_arrange=[0] #overfit

for epoch in range(tot_epoch):
    random.shuffle(batch_arrange)
    for batch_s in batch_arrange:
        optimizer.zero_grad()
        x0_batch, x1_batch, y_batch = generate_batch(x0, x1, Y, batch_size, batch_s)
        y_out = model(x0_batch, x1_batch)
        loss = criterion(y_out, y_batch)
        loss.backward()
        optimizer.step()
    if epoch%100==0:
        print("epoch ", epoch, ": current loss ", loss, sep="")
    if epoch%1000==1:
        torch.save(model.state_dict(), 'models/attention_epoch_' + str(epoch) + '.torch')
