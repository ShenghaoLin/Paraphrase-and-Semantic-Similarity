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
TRAIN_DATA_PATH = '../data/train_data_tree_'+str(d_model)+'d.pkl'
f=open(TRAIN_DATA_PATH, "rb")
train_x0=pickle.load(f)
train_x1=pickle.load(f)
train_Y=pickle.load(f)
train_x0_r=pickle.load(f)
train_x1_r=pickle.load(f)
f.close()
train_Y = train_Y.long()
print(train_Y[:10])

VAL_DATA_PATH = '../data/dev_data_tree_'+str(d_model)+'d.pkl'
f=open(VAL_DATA_PATH, "rb")
val_x0=pickle.load(f)
val_x1=pickle.load(f)
val_Y=pickle.load(f)
val_x0_r=pickle.load(f)
val_x1_r=pickle.load(f)
f.close()
val_Y = val_Y.long()
print(val_Y[:10])

d_len=train_x0.size()[0]
s_len=train_x0.size()[1]

def make_model(d_model):
    # This was important from their code. 
    # Initialize parameters with Glorot / fan_avg.
    model = SimilarityTreeLSTM(d_model)
    
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform(p)
    return model

def compute_acc(val_x0, val_x1, val_Y, val_x0_r, val_x1_r, model):
    cor=0
    for j in range(len(val_x0)):
        y_out = model(val_x0_r[j], val_x0[j].to(device), val_x1_r[j], val_x1[j].to(device))
        y_out = F.log_softmax(y_out, dim=1)
        y_out = torch.argmax(y_out, dim=1)
        temp = y_out[0].cpu()
        if temp<3 and val_Y[j]<3:
            cor+=1;
        elif temp>=3 and val_Y[j]>=3:
            cor+=1;
    print("current accuracy is :", cor/len(val_x0))
    return cor/len(val_x0)

model = make_model(d_model).to(device)
criterion = nn.CrossEntropyLoss()
learning_rate=1e-3
reg=1e-7
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=reg)
tot_epoch=200
batch_size=100
batch_arrange = [i for i in range(0, d_len)]
loss_history=[]
val_acc_history=[]

torch.cuda.empty_cache()

total_loss = 0.0
max_acc = 0.0

for epoch in range(tot_epoch):
    total_loss = 0.0
    random.shuffle(batch_arrange)
    for (idx, i) in enumerate(batch_arrange):
        y_out = model(train_x0_r[i], train_x0[i].to(device), train_x1_r[i], train_x1[i].to(device))
        loss = criterion(y_out, train_Y[i].unsqueeze_(0).to(device))
        loss.backward()
        total_loss += loss.item()
        if idx % batch_size == 0 and idx > 0:
            optimizer.step()
            optimizer.zero_grad()
            loss_history.append(total_loss)
    print("epoch ", epoch, ": current loss ", total_loss, sep="")
    cur_acc = compute_acc(val_x0, val_x1, val_Y, val_x0_r, val_x1_r, model)
    val_acc_history.append(cur_acc)
    if cur_acc > max_acc:
        max_acc=cur_acc
        torch.save(model.state_dict(), 'models/treelstm_' + str(d_model) +'d_epoch_' + str(epoch) + '.torch')
    elif epoch%idx==0:
        torch.save(model.state_dict(), 'models/treelstm_' + str(d_model) +'d_epoch_' + str(epoch) + '.torch')
        

f=open('treelstm_'+str(d_model)+'d.his', 'wb')
pickle.dump(loss_history, f)
pickle.dump(val_acc_history, f)
f.close()
