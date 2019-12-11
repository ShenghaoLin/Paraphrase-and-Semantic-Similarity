import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import copy
import math

from model_cls import *

import random
import sys
import pickle
import argparse

device = torch.device(0 if torch.cuda.is_available() else "cpu")

d_model=50
TRAIN_DATA_PATH = '../data/train_data_'+str(d_model)+'d.pkl'
f=open(TRAIN_DATA_PATH, "rb")
train_x0=pickle.load(f)
train_x1=pickle.load(f)
train_Y=pickle.load(f)
f.close()

VAL_DATA_PATH = '../data/dev_data_'+str(d_model)+'d.pkl'
f=open(VAL_DATA_PATH, "rb")
val_x0=pickle.load(f)
val_x1=pickle.load(f)
val_Y=pickle.load(f)
f.close()

d_len=train_x0.size()[0]
s_len=train_x0.size()[1]
print(train_x0.size())
print(train_x1.size())
print(train_Y.size())

    
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

def compute_acc(val_x0, val_x1, val_Y, model, reg=False):
    val_b=500
    cor=0
    for i in range(0, len(val_Y), val_b):
        val_x0_batch, val_x1_batch, val_Y_batch=generate_batch(val_x0, val_x1, val_Y, val_b, i)
        y_out = model(val_x0_batch, val_x1_batch)
        if not reg:
            y_out = F.log_softmax(y_out, dim=1)
            y_out = torch.argmax(y_out, dim=1)
        for i in range(len(y_out)):
            if reg:
                if y_out[i]<0.6 and val_Y_batch[i]<0.6:
                    cor+=1;
                elif y_out[i]>=0.6 and val_Y_batch[i]>=0.6:
                    cor+=1;
            else:
                if y_out[i]<3 and val_Y_batch[i]<3:
                    cor+=1;
                elif y_out[i]>=3 and val_Y_batch[i]>=3:
                    cor+=1;
                
    print("current accuracy is :", cor/len(val_Y))
    return cor/len(val_Y)

model = make_model(d_model, s_len).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate=5e-4
reg=1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
tot_epoch=100
batch_size=100
# batch_arrange=[0] #overfit
batch_arrange = [i for i in range(0, d_len, batch_size)]
loss_history=[]
val_acc_history=[]

torch.cuda.empty_cache()
max_acc=0.0

for epoch in range(tot_epoch):
    random.shuffle(batch_arrange)
    for start_i in batch_arrange:
        x0_batch, x1_batch, Y_batch = generate_batch(train_x0, train_x1, train_Y, batch_size, start_i)
        optimizer.zero_grad()
        y_out = model(x0_batch, x1_batch)
        loss = criterion(y_out, Y_batch)
        loss.backward()
        optimizer.step()
        loss_history.append(loss)
    print("epoch ", epoch, ": current loss ", loss, sep="")
    cur_acc=compute_acc(val_x0, val_x1, val_Y, model, reg=False)
    val_acc_history.append(cur_acc)
    if cur_acc>max_acc:
        max_acc=cur_acc
        torch.save(model.state_dict(), 'models/angle_attention_' + str(d_model) +'d_epoch_' + str(epoch) + '.torch')
    elif epoch%20==0:
        torch.save(model.state_dict(), 'models/angle_attention_' + str(d_model) +'d_epoch_' + str(epoch) + '.torch')        

f=open(str(d_model)+'d_cls.his', 'wb')
pickle.dump(loss_history, f)
pickle.dump(val_acc_history, f)
f.close()
