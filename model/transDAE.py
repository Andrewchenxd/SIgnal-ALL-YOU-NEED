import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math

class transDAE(nn.Module):
    def __init__(self, numclass,embeddingdim=32):
        super(transDAE, self).__init__()
        self.embed=nn.Sequential(nn.Linear(2,embeddingdim),
                                 nn.Dropout(0.2))
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embeddingdim, nhead=4)

        self.trans=nn.TransformerEncoder(self.encoder_layer, num_layers=3)

        self.fc1 = nn.Sequential(nn.Linear(in_features=32, out_features=32),
                                 nn.BatchNorm1d(32),
                                 )
        self.fc2 = nn.Sequential(nn.Linear(in_features=32, out_features=16),
                                 nn.BatchNorm1d(16),
                                )
        self.fc3 = nn.Linear(in_features=16, out_features=numclass)
        self.fc4 = nn.Linear(in_features=32, out_features=2)

    def forward(self, y):

        y = y.transpose(1, 2)
        y=self.embed(y)
        y = self.trans(y)
        x = self.fc4(y)
        x = x.transpose(1, 2)
        y = y[:, -1, :]
        y=self.fc1(y)
        y = self.fc2(y)
        y = self.fc3(y)
        return x,y


# net1= transDAE(10)
# a=torch.randn((2,1,128,128))
# b=torch.randn((3,2,128))
# c=torch.randn((2,1,17))
#
# net1(b)