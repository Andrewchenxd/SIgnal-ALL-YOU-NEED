import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Bottleneck(nn.Module):
    # 每个stage维度中扩展的倍数
    extention = 4

    def __init__(self, inplanes, planes, stride, downsample=None):
        '''

        :param inplanes: 输入block的之前的通道数
        :param planes: 在block中间处理的时候的通道数
                planes*self.extention:输出的维度
        :param stride:
        :param downsample:
        '''
        super(Bottleneck, self).__init__()

        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, stride=stride, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)

        self.conv2 = nn.Conv1d(planes, planes, kernel_size=7, stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)

        self.conv3 = nn.Conv1d(planes, planes * self.extention, kernel_size=1, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * self.extention)

        self.relu = nn.ReLU(inplace=True)

        # 判断残差有没有卷积
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 参差数据
        residual = x

        # 卷积操作
        out1 = self.conv1(x)
        out2 = self.bn1(out1)
        out3 = self.relu(out2)

        out4 = self.conv2(out3)
        out5 = self.bn2(out4)
        out6 = self.relu(out5)

        out7 = self.conv3(out6)
        out8 = self.bn3(out7)
        out9 = self.relu(out8)

        # 是否直连（如果Indentity blobk就是直连；如果Conv2 Block就需要对残差边就行卷积，改变通道数和size
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差部分和卷积部分相加
        out10 = out9 + residual
        out11 = self.relu(out10)

        return out11


class Resnet1d(nn.Module):
    def __init__(self,  block=Bottleneck, layers=[3, 4, 23, 3]):
        # inplane=当前的fm的通道数
        self.inplane = 64
        super(Resnet1d, self).__init__()

        # 参数
        self.block = block
        self.layers = layers

        # stem的网络层
        self.conv1 = nn.Conv1d(1, self.inplane, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplane)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # 64,128,256,512指的是扩大4倍之前的维度，即Identity Block中间的维度
        self.stage1 = self.make_layer(self.block, 64, layers[0], stride=1)
        self.stage2 = self.make_layer(self.block, 128, layers[1], stride=2)
        self.stage3 = self.make_layer(self.block, 256, layers[2], stride=2)
        self.stage4 = self.make_layer(self.block, 512, layers[3], stride=2)
        self.gap = nn.AdaptiveAvgPool1d(1)


    def forward(self, x):


        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        # block部分
        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.gap(out)
        out = out.view(out.size(0), -1)
        return out

    def make_layer(self, block, plane, block_num, stride=1):
        '''
        :param block: block模板
        :param plane: 每个模块中间运算的维度，一般等于输出维度/4
        :param block_num: 重复次数
        :param stride: 步长
        :return:
        '''
        block_list = []
        # 先计算要不要加downsample
        downsample = None
        if (stride != 1 or self.inplane != plane * block.extention):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplane, plane * block.extention, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm1d(plane * block.extention)
            )

        # Conv Block输入和输出的维度（通道数和size）是不一样的，所以不能连续串联，他的作用是改变网络的维度
        # Identity Block 输入维度和输出（通道数和size）相同，可以直接串联，用于加深网络
        # Conv_block
        conv_block = block(self.inplane, plane, stride=stride, downsample=downsample)
        block_list.append(conv_block)
        self.inplane = plane * block.extention

        # Identity Block
        for i in range(1, block_num):
            block_list.append(block(self.inplane, plane, stride=1))

        return nn.Sequential(*block_list)


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.heads = nn.Sequential(
            nn.Linear(dim, num_classes),
        )

    def forward(self, img,y,z):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        x = x.view(x.size(0), -1)
        x = self.heads(x)
        return x

if __name__ == '__main__':
    img = torch.randn(1, 1, 128, 128)
    sgn= torch.randn(1, 1, 2999)
    # resnet=Resnet1d()
    # resnet(sgn)#2048
    model = ViT(
            image_size = 128,
            channels=1,
            patch_size = 2,
            num_classes = 10,
            dim = 512,
            depth = 6,
            heads = 16,
            mlp_dim = 1024,
            dropout = 0.2,
            emb_dropout = 0.2
        )
    preds = model(img,img,img)

