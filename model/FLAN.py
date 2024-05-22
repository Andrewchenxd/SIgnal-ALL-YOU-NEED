import torch
from torch import nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    #每个stage维度中扩展的倍数
    extention=4
    def __init__(self,inplanes,planes,stride,downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1=nn.Conv1d(inplanes,planes,kernel_size=1,stride=stride,bias=False)
        self.bn1=nn.BatchNorm1d(planes)

        self.conv2=nn.Conv1d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2=nn.BatchNorm1d(planes)

        self.conv3=nn.Conv1d(planes,planes*self.extention,kernel_size=1,stride=1,bias=False)
        self.bn3=nn.BatchNorm1d(planes*self.extention)

        self.relu=nn.ReLU(inplace=True)

        #判断残差有没有卷积
        self.downsample=downsample
        self.stride=stride

    def forward(self,x):
        #参差数据
        residual=x

        #卷积操作
        out1=self.conv1(x)
        out2=self.bn1(out1)
        out3=self.relu(out2)

        out4=self.conv2(out3)
        out5=self.bn2(out4)
        out6=self.relu(out5)

        out7=self.conv3(out6)
        out8=self.bn3(out7)
        out9=self.relu(out8)

        #是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual=self.downsample(x)

        #将残差部分和卷积部分相加
        out10=out9+residual
        out11=self.relu(out10)

        return out11

class Multi_spectral_attention_mechanism(nn.Module):
    '''
    input:(B,C,L)
    功能:实现L中每个频域点的选择
    '''
    def __init__(self, input_channel=256,input_L=32):
        super(Multi_spectral_attention_mechanism, self).__init__()
        self.cnn = nn.Sequential(nn.Conv1d(input_channel, input_channel, kernel_size=1, stride=1, padding=0, bias=False),
                                 nn.BatchNorm1d(input_channel))
        self.fc=nn.Sequential(nn.Linear(input_L,input_L//2),
                              nn.Dropout(0.2),
                              nn.Linear(input_L//2,1))
    def forward(self, x):
        DCT =  torch.abs(torch.fft.fft2(x))
        DCT_res=self.cnn(DCT)
        score=torch.nn.functional.gumbel_softmax(DCT_res,tau=0.01)
        DCT=DCT*score
        attn=torch.sigmoid(self.fc(DCT))
        x=x*attn
        return x

class FLAN(nn.Module):
    def __init__(self, num_classes,block=Bottleneck,layers=[3,4,6,3],input_L=[32,16,8,4]):
        #inplane=当前的fm的通道数
        self.inplane=64
        super(FLAN, self).__init__()

        #参数
        self.block=block
        self.layers=layers

        #stem的网络层
        self.conv1=nn.Conv1d(2,self.inplane,kernel_size=7,stride=2,padding=3,bias=False)
        self.bn1=nn.BatchNorm1d(self.inplane)
        self.relu=nn.ReLU()
        self.maxpool=nn.MaxPool1d(kernel_size=3,stride=2,padding=1)

        #64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1=self.make_layer(self.block,64,layers[0],stride=1,input_L=input_L[0])
        self.fre_attn1 = Multi_spectral_attention_mechanism(input_channel=256,input_L=32)
        self.stage2=self.make_layer(self.block,128,layers[1],stride=2,input_L=input_L[1])
        self.fre_attn2 = Multi_spectral_attention_mechanism(input_channel=512,input_L=16)
        self.stage3=self.make_layer(self.block,256,layers[2],stride=2,input_L=input_L[2])
        self.fre_attn3 = Multi_spectral_attention_mechanism(input_channel=1024,input_L=8)
        self.stage4=self.make_layer(self.block,512,layers[3],stride=2,input_L=input_L[3])
        self.fre_attn4 = Multi_spectral_attention_mechanism(input_channel=2048,input_L=4)
        self.gap=nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self,y):
        y = y.squeeze(1)
        #stem部分：conv+bn+maxpool
        out=self.conv1(y)
        out=self.bn1(out)
        out=self.relu(out)
        out=self.maxpool(out)

        #block部分
        out=self.stage1(out)
        # out=self.fre_attn1(out)
        out=self.stage2(out)
        # out = self.fre_attn2(out)
        out=self.stage3(out)
        # out = self.fre_attn3(out)
        out=self.stage4(out)

        out = self.gap(out)
        out=out.view(out.size(0), -1)
        out=self.classifier(out)

        return out

    def make_layer(self,block,plane,block_num,input_L,stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list=[]
        #先计算要不要加downsample
        downsample=None
        if(stride!=1 or self.inplane!=plane*block.extention):
            downsample=nn.Sequential(
                nn.Conv1d(self.inplane,plane*block.extention,stride=stride,kernel_size=1,bias=False),
                nn.BatchNorm1d(plane*block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        #Conv_block
        conv_block=block(self.inplane,plane,stride=stride,downsample=downsample)
        block_list.append(conv_block)

        block_list.append(Multi_spectral_attention_mechanism(input_channel=plane*4,input_L=input_L))

        self.inplane=plane*block.extention

        #Identity Block
        for i in range(1,block_num):
            block_list.append(block(self.inplane,plane,stride=1))

        return nn.Sequential(*block_list)

if __name__ == '__main__':
    import time
    net1= FLAN(num_classes=10).cuda()
    total = 32
    a = torch.randn((total, 1, 128, 128)).cuda()
    b = torch.randn((total, 2, 128)).cuda()
    c = torch.randn((total, 1, 17)).cuda()
    # net1(a, b, c)
    for _ in range(100):
        net1(b)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(100):
        net1(b)
    torch.cuda.synchronize()
    end = time.perf_counter()

    print('{} ms'.format((end - begin) / (100 * total) * 1000))


    def count_parameters_in_MB(model):
        return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


    num_params = count_parameters_in_MB(net1)
    print(f'Number of parameters: {num_params}')