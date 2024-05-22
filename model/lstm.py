import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math
class lstm2(nn.Module):

    def __init__(self, output_size=128,numclass=10):
        super(lstm2, self).__init__()
        # lstm的输入 #batch,seq_len(词的长度）, input_size(词的维数）
        self.rnn1 = nn.LSTM(input_size=2, hidden_size=output_size, batch_first=True)
        self.rnn2 = nn.LSTM(input_size=output_size, hidden_size=output_size, batch_first=True)
        self.fc=nn.Linear(output_size,numclass)

    def forward(self, x,y,z):
        self.rnn1.flatten_parameters()
        self.rnn2.flatten_parameters()
        y = y.transpose(1, 2)

        out, (hidden, cell) = self.rnn1(
            y)  # y.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        out, (hidden, cell) = self.rnn2(out)
        out =out[:, -1, :]
        out = out .view(out .size(0), -1)
        out=self.fc(out)
        return out

# net1= lstm2(128,10)
# a=torch.randn((2,1,128,128))
# b=torch.randn((2,2,128))
# c=torch.randn((2,1,17))
#
# net1(a,b,c)











