import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from utils.signeltoimage import *
import torch.fft
import math
torch.manual_seed(42)
torch.cuda.manual_seed(42)
d_model = 64
d_ff = 32
d_q = 1
d_k = 1
d_v = 1
n_heads = 2112

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

class SKConv(nn.Module):
    def __init__(self, channels, branches=2, groups=32, reduce=16, stride=1, len=32):
        super(SKConv, self).__init__()
        len = max(channels // reduce, len)
        self.convs = nn.ModuleList([])
        for i in range(branches):
            self.convs.append(nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, stride=stride, padding=1 + i, dilation=1 + i,
                          groups=groups, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            ))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(channels, len, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(len),
            nn.ReLU(inplace=True)
        )
        self.fcs = nn.ModuleList([])
        for i in range(branches):
            self.fcs.append(
                nn.Conv2d(len, channels, kernel_size=1, stride=1)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = [conv(x) for conv in self.convs]
        x = torch.stack(x, dim=1)
        attention = torch.sum(x, dim=1)
        attention = self.gap(attention)
        attention = self.fc(attention)
        attention = [fc(attention) for fc in self.fcs]
        attention = torch.stack(attention, dim=1)
        attention = self.softmax(attention)
        x = torch.sum(x * attention, dim=1)
        return x

class SKUnit(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, branches=2, group=32, reduce=16, stride=1, len=32):
        super(SKUnit, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        self.conv2 = SKConv(mid_channels, branches=branches, groups=group, reduce=reduce, stride=stride, len=len)

        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        if in_channels == out_channels:  # when dim not change, input_features could be added directly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, input_features should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        residual = self.shortcut(residual)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x += residual
        return self.relu(x)

class sknetdrop(nn.Module):
    def __init__(self, num_classes, hidden_size,cutmixsize, num_block_lists=[3, 4, 6, 3]):
        super(sknetdrop, self).__init__()
        self.cutmixsize=cutmixsize
        self.basic_conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

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
        self.stage_1 = self._make_layer(64, 128, 256, nums_block=num_block_lists[0], stride=1)
        self.stage_2 = self._make_layer(256, 256, 512, nums_block=num_block_lists[1], stride=2)
        self.stage_3 = self._make_layer(512, 512, 1024, nums_block=num_block_lists[2], stride=2)
        self.stage_4 = self._make_layer(1024, 1024, 2048, nums_block=num_block_lists[3], stride=2)
        self.sgncov3 = nn.Sequential(
            nn.Conv1d(1, hidden_size, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )

        self.emb = nn.Linear(1, d_model)
        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.2)
        self.drop3 = nn.Dropout(0.2)
        self.op_emb_t = nn.Linear(2 * hidden_size, 1)
        self.op_emb_s = nn.Linear(2 * hidden_size, 1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gap2 = nn.AdaptiveAvgPool1d(1)
        self.gap3 = nn.AdaptiveAvgPool1d(1)
        self.attention=MultiHeadAttention()
        self.classifier = nn.Linear(1024, num_classes)
        self.fc = nn.Linear(in_features=2112, out_features=1024)
        self.drop=nn.Dropout(0.2)
        self.softMax=nn.Softmax(dim=0)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)


    def _make_layer(self, in_channels, mid_channels, out_channels, nums_block, stride=1):
        layers = [SKUnit(in_channels, mid_channels, out_channels, stride=stride)]
        for _ in range(1, nums_block):
            layers.append(SKUnit(out_channels, mid_channels, out_channels))
        return nn.Sequential(*layers)
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

    def forward(self, x,z):
        x = self.basic_conv(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.stage_4(x)
        x = self.gap(x)

        z = self.sgncov3(z)
        z=self.gap2(z)
        x = x.view(x.size(0), -1)
        z = z.view(z.size(0), -1)
        x = torch.cat((x,z), 1)
        x = self.emb(x.view(-1, x.shape[1], 1))
        x=self.drop3(x)
        contex=self.attention(x,x,x)
        x=self.gap3(contex)
        x = x.view(x.size(0), -1)
        x=self.fc(x)
        x=self.drop(x)
        x = self.classifier(x)
        return x


# net1= sknetdrop(10,64,4)
# a=torch.randn((2,1,128,128))
# c=torch.randn((2,1,17))
#
# net1(a,c)