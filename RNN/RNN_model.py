import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNN_model(nn.Module):

    def __init__(self, intput_size, hidden_size, class_num, num_layers=1, dropout=0):
        super(RNN_model, self).__init__()
        self.rnn = nn.GRU(intput_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.linear = nn.Linear(hidden_size, class_num)
        nn.init.xavier_uniform_(self.linear.weight)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_0, x_1):
        x_0, _ = self.rnn(x_0)
        x_1, _ = self.rnn(x_1)
        x = torch.mul(x_0[:, -1, :], x_1[:, -1, :])
        # print(x)

        x = self.linear(x)
        x = self.softmax(x)
        # print(x.size())
        return x