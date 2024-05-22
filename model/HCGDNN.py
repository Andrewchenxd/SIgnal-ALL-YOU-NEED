import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math

class HCGDNN(nn.Module):
    def __init__(self, numclass):
        super(HCGDNN, self).__init__()
        self.conv_net = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(64),
            nn.Dropout(0.2),
            nn.Conv2d(64, 128, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
            nn.Conv2d(128, 128, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(1,2)),
            nn.BatchNorm2d(128),
            nn.Dropout(0.2),
        )
        self.gru1 = nn.GRU(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.2)
        self.gru2 = nn.GRU(input_size=128, hidden_size=64, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(in_features=128, out_features=1024),
                                 nn.Dropout(0.2),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1024, out_features=numclass)
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=128, out_features=1024),
                                 nn.Dropout(0.2),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1024, out_features=numclass)
                                 )
        self.fc3 = nn.Sequential(nn.Linear(in_features=128, out_features=1024),#23680
                                 nn.Dropout(0.2),
                                 nn.ReLU(),
                                 nn.Linear(in_features=1024, out_features=numclass),
                                 )
    def forward(self,y):
        # y = y.unsqueeze(2)
        y=torch.transpose(y,1,2)
        c_fea = self.conv_net(y)

        c_fea = torch.squeeze(c_fea, dim=2)
        c_fea = torch.transpose(c_fea, 1, 2)
        g_fea1, _ = self.gru1(c_fea)
        fea = self.dropout(g_fea1)
        g_fea2, _ = self.gru2(fea)
        c_fea =torch.sum(c_fea,dim=1)
        g_fea1=torch.sum(g_fea1,dim=1)
        g_fea2 = torch.sum(g_fea2, dim=1)
        # c_fea = c_fea.contiguous().view(c_fea.size()[0], -1)
        # g_fea1 = g_fea1.contiguous().view(g_fea1.size()[0], -1)
        # g_fea2 = g_fea2.contiguous().view(g_fea2.size()[0], -1)
        c_fea =self.fc1(c_fea)
        g_fea1=self.fc2(g_fea1)
        g_fea2 = self.fc3(g_fea2)
        return c_fea,g_fea1,g_fea2
if __name__ == '__main__':
# #
    net1= HCGDNN(24)
    len=128
    a=torch.randn((2,1,128,128))
    b=torch.randn((3,1,2,len))
    c=torch.randn((2,1,17))

    net1(b)

    # def count_parameters_in_MB(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad) * 4 / (1024 ** 2)
    #
    # num_params = count_parameters_in_MB(net1)
    # print(f'Number of parameters: {num_params}')