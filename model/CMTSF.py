import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math
torch.manual_seed(114514)
torch.cuda.manual_seed(114514)
d_model = 64
d_ff = 32
d_q = 1
d_k = 1
d_v = 1
n_heads = 512

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaledDotProductAttention, self).__init__()

    def forward(self, Q, K, V):
        K=K/(K.norm(p="fro",dim=2).unsqueeze(dim=3))
        Q = Q / (Q.norm(p="fro", dim=2).unsqueeze(dim=3))
        K_TV = torch.matmul(K.transpose(-1, -2),V) / np.sqrt(d_k)
        context = torch.matmul(Q,K_TV)
        return context

class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_q * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)


    def forward(self, input_Q, input_K, input_V):
        batch_size = input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_q).transpose(1, 2)
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)
        context = ScaledDotProductAttention()(Q, K, V)
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v)

        return  context

class cmtsf(nn.Module):
    def __init__(self, num_classes, hidden_size,cutmixsize, num_block_lists=[3, 4, 6, 3],drop=0.2,in_features=256):
        super(cmtsf, self).__init__()
        self.cutmixsize=cutmixsize

        self.sgncovt = nn.Sequential(
            nn.Conv1d(2, hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(2 * hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder_layer_t = nn.TransformerEncoderLayer(d_model=2 * hidden_size, nhead=4)
        self.transformer_encoder_t = nn.TransformerEncoder(self.encoder_layer_t, num_layers=3)
        self.sgncovs  = nn.Sequential(
            nn.Conv1d(2, hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_size, 2*hidden_size, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(2*hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        self.encoder_layer_s = nn.TransformerEncoderLayer(d_model=2 * hidden_size, nhead=4)
        self.transformer_encoder_s = nn.TransformerEncoder(self.encoder_layer_s, num_layers=3)
        self.cutcovt=self.cutmix_make_layer(1,hidden_size,self.cutmixsize)
        self.cutcovs = self.cutmix_make_layer(1, hidden_size, self.cutmixsize)

        self.emb = nn.Linear(1, d_model)
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.drop3 = nn.Dropout(drop)
        self.op_emb_t = nn.Linear(2 * hidden_size, 1)
        self.op_emb_s = nn.Linear(2 * hidden_size, 1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap3 = nn.AdaptiveAvgPool1d(1)
        self.attention=MultiHeadAttention()
        self.classifier = nn.Linear(128, num_classes)
        self.fc = nn.Linear(in_features=in_features, out_features=128)
        self.drop=nn.Dropout(drop)
        self.softMax=nn.Softmax(dim=0)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def cutmix_make_layer(self,in_channels, hidden_size,cutsize):
        layers = [nn.Conv2d(in_channels, int(hidden_size/2), kernel_size=3, stride=2, padding=1)]
        for i in range(1, cutsize):
            layers.append(nn.Conv2d(int(hidden_size/2)*i,int(hidden_size/2)*(i+1), kernel_size=3, stride=2, padding=1))
        return nn.ModuleList(layers)

    # 默认基是16
    def cutmix(self,fbt, fbs, cutsize, id):
        len = fbt.shape[3]
        wid = fbt.shape[2]
        base = 16
        temp_mask = np.zeros([base, len, wid])
        base_mask = np.zeros([base, len, wid])
        mask = np.zeros([len, wid])
        num = math.floor(base / cutsize)
        xx = math.ceil(len / cutsize)
        k = 0
        for i in range(3):
            i = i + 1
            for j in range(i):
                temp_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)] = 1
                base_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)] = np.tril(
                    temp_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)])
                k = k + 1
                temp_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)] = 1
                base_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)] = np.triu(
                    temp_mask[k, xx * i:xx * (i + 1), xx * j:xx * (j + 1)], 1)
                k = k + 1
        for i in range(4):
            temp_mask[k, xx * i:xx * (i + 1), xx * i:xx * (i + 1)] = 1
            base_mask[k, xx * i:xx * (i + 1), xx * i:xx * (i + 1)] = np.tril(
                temp_mask[k, xx * i:xx * (i + 1), xx * i:xx * (i + 1)])
            k = k + 1
        if id != (cutsize - 1):
            for i in range(num):
                mask = mask + base_mask[i + id * num, :, :]
        else:
            for i in range(base - num * (cutsize - 1)):
                mask = mask + base_mask[i + id * num, :, :]
        mask = np.expand_dims(mask, axis=0)
        mask = np.expand_dims(mask, axis=0)

        mask = torch.tensor(mask, dtype=torch.float32,device=fbt.device)
        tem_t = fbt * mask
        tem_s = fbs * mask
        fbt = fbt * (1 - mask) + tem_t
        fbs = fbs * (1 - mask) + tem_s
        return fbt, fbs

    def forward(self, x, y,z):

        f = abs(torch.fft.fft(y))
        f = self.sgncovs(f)
        f = f.transpose(1, 2)
        f = self.transformer_encoder_s(f)
        f=self.op_emb_s(f)
        f=self.drop1(f)
        fs_T = f.permute(0, 2, 1)
        fbs=f*fs_T
        fbs = fbs.unsqueeze(1)

        y=self.sgncovt(y)
        y = y.transpose(1, 2)
        y = self.transformer_encoder_t(y)
        y = self.op_emb_t(y)
        y=self.drop2(y)
        ft_T = y.permute(0, 2, 1)
        fbt = y * ft_T
        fbt = fbt.unsqueeze(1)
        for i in range(self.cutmixsize):
            fbt, fbs = self.cutmix(fbt, fbs, self.cutmixsize,i)
            fbt = self.cutcovt[i](fbt)
            fbs = self.cutcovs[i](fbs)
        y=torch.cat((fbt,fbs), 1)
        y = self.gap1(y)
        y = y.view(y.size(0), -1)
        y = self.emb(y.view(-1, y.shape[1], 1))
        y=self.drop3(y)
        contex=self.attention(y,y,y)
        y=self.gap3(contex)
        y = y.view(y.size(0), -1)
        y=self.fc(y)
        y=self.drop(y)
        y = self.classifier(y)
        return y

#
# net1= cmtsf(10,64,8,in_features=512)
# a=torch.randn((2,1,128,128))
# b=torch.randn((2,2,128))
# c=torch.randn((2,1,17))
#
# net1(a,b,c)