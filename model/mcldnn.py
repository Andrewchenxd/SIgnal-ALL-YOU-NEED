import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
# from utils.signeltoimage import *
import torch.fft
import math
class lstm(nn.Module):

    def __init__(self, input_size,output_size):
        super(lstm, self).__init__()
        # lstm的输入 #batch,seq_len(词的长度）, input_size(词的维数）
        self.rnn1 = nn.LSTM(input_size=input_size, hidden_size=output_size,batch_first=True)
        self.rnn2 = nn.LSTM(input_size=output_size, hidden_size=output_size, batch_first=True)

    def forward(self, x):
        # self.rnn.flatten_parameters()
        # x.shape : batch,seq_len,hidden_size , hn.shape and cn.shape : num_layes * direction_numbers,batch,hidden_size
        out, (hidden, cell) = self.rnn1(x)
        out, (hidden, cell) = self.rnn2(out)
        out =out[:, -1, :]

        return out

class mcldnn(nn.Module):
    def __init__(self, num_classes):
        super(mcldnn, self).__init__()
        self.conv1=nn.Sequential(nn.Conv2d(1, 50, kernel_size=(1,7), stride=1,padding=(0,3) ,bias=True),
                                )
        self.conv2 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=7, stride=1,padding=3, bias=True),
                                )
        self.conv3 = nn.Sequential(nn.Conv1d(1, 50, kernel_size=7, stride=1,padding=3, bias=True),
                                   )
        self.conv4 = nn.Sequential(nn.Conv2d(50, 50, kernel_size=(1,7), stride=1,padding=(0,3), bias=True),
                                   )

        self.conv5 = nn.Sequential(nn.Conv2d(100, 100, kernel_size=(2, 5), stride=1, bias=True),
                               )

        self.encoder_layer_t = lstm(100,128)

        self.fc1 = nn.Sequential(nn.Linear(in_features=128, out_features=128),
                                 nn.SELU(inplace=True),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=128, out_features=128),
                                 nn.SELU(inplace=True),
                                )
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

    def forward(self, y):
        # y = y.unsqueeze(1)
        I=y[:,:,0,:]
        Q = y[:,:, 1, :]
        y=self.conv1(y)
        I=self.conv2(I)
        I = I.unsqueeze(2)
        Q = self.conv3(Q)
        Q = Q.unsqueeze(2)
        IQ = torch.cat((I, Q), 2)

        IQ=self.conv4(IQ)
        y=torch.cat((IQ, y), 1)
        y=self.conv5(y)
        y = y.squeeze(2)
        y = y.transpose(1, 2)
        y = self.encoder_layer_t(y)
        # y = y.permute(1, 2, 0)
        y = y.view(y.size(0), -1)
        y=self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return y


if __name__ == '__main__':
    print(mcldnn(10))
    net1= mcldnn(10)
    a=torch.randn((2,1,128,128))
    b=torch.randn((3,2,128))
    c=torch.randn((2,1,17))

    net1(b)