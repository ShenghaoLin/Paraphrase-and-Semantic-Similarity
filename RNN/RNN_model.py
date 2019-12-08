import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RNN_model(nn.Module):

    def __init__(self, intput_size, hidden_size, class_num, num_layers=5, dropout=0, skip_connection=False):
        super(RNN_model, self).__init__()
        self.rnn = nn.GRU(intput_size, hidden_size, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.skip_connection = skip_connection
        if skip_connection:
            self.rnn_1 = nn.GRU(intput_size, intput_size, num_layers=1, dropout=dropout, batch_first=True)
            self.rnn_2 = nn.GRU(intput_size, intput_size, num_layers=1, dropout=dropout, batch_first=True)
            self.rnn_3 = nn.GRU(intput_size, intput_size, num_layers=1, dropout=dropout, batch_first=True)
            self.rnn_4 = nn.GRU(intput_size, intput_size, num_layers=1, dropout=dropout, batch_first=True)
            self.rnn_5 = nn.GRU(intput_size, intput_size, num_layers=1, dropout=dropout, batch_first=True)
            self.linear_0 = nn.Linear(intput_size, hidden_size, class_num)
        self.linear_1 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.linear_2 = nn.Linear(hidden_size * 2, class_num)
        nn.init.xavier_uniform_(self.linear_1.weight)
        nn.init.xavier_uniform_(self.linear_2.weight)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_0, x_1):
        if self.skip_connection:
            tmp, _ = self.rnn_1(x_0)
            x_0 = tmp + x_0
            tmp, _ = self.rnn_2(x_0)
            x_0 = tmp + x_0
            tmp, _ = self.rnn_3(x_0)
            x_0 = tmp + x_0
            tmp, _ = self.rnn_4(x_0)
            x_0 = tmp + x_0
            tmp, _ = self.rnn_5(x_0)
            x_0 = tmp + x_0
            tmp, _ = self.rnn_1(x_1)
            x_1 = tmp + x_1
            tmp, _ = self.rnn_2(x_1)
            x_1 = tmp + x_1
            tmp, _ = self.rnn_3(x_1)
            x_1 = tmp + x_1
            tmp, _ = self.rnn_4(x_1)
            x_1 = tmp + x_1
            tmp, _ = self.rnn_5(x_1)
            x_1 = tmp + x_1

            x_0 = self.relu(self.linear_0(x_0))
            x_1 = self.relu(self.linear_0(x_1))

        else:
            x_0, _ = self.rnn(x_0)
            x_1, _ = self.rnn(x_1)

        x_mul = torch.mul(x_0[:, -1, :], x_1[:, -1, :])
        x_add = torch.abs(torch.add(x_0[:, -1, :], -x_1[:, -1, :]))
            
        x = torch.cat((x_mul, x_add), dim=1)

        x = self.relu(self.linear_1(x))
        x = torch.nn.functional.sigmoid(self.linear_2(x))

        return x
