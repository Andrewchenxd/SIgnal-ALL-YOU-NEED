import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math
class lstm(nn.Module):

    def __init__(self, input_size,output_size):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len(词的长度）, input_size(词的维数）
        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=output_size,batch_first=True)
        self.rnn2 = nn.LSTM(input_size=output_size, hidden_size=output_size, batch_first=True)

    def forward(self, x):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        out, (hidden, cell) = self.rnn1(x)
        out1, (hidden, cell) = self.rnn2(out)
        out =out1[:, -1, :]

        return out1,out

class DAE(nn.Module):
    def __init__(self, numclass):
        super(DAE, self).__init__()
        self.lstm=lstm(2,32)

        self.fc1 = nn.Sequential(nn.Linear(in_features=32, out_features=32),
                                 nn.BatchNorm1d(32),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=32, out_features=16),
                                 nn.BatchNorm1d(16),
                                )
        self.fc3 = nn.Linear(in_features=16, out_features=numclass)
        self.fc4 = nn.Linear(in_features=32, out_features=2)

    def forward(self, y):
        y=torch.squeeze(y,1)
        y = y.transpose(1, 2)
        x,y = self.lstm(y)
        # y = y.permute(1, 2, 0)
        x = self.fc4(x)
        x = x.transpose(1, 2)
        y=self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return x,y

#
# net1= DAE(10)
# a=torch.randn((2,1,128,128))
# b=torch.randn((3,2,128))
# c=torch.randn((2,1,17))
# net1(b)