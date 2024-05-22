import numpy as np
import torch.fft
from einops import rearrange
from pwvdswin_ViT import *
from utils.pos_embed import get_2d_sincos_pos_embed
from functools import partial
from .SIT import ConvolutionalEncoding, PositionalEncoding,TransformerEncoder,TemporalConvolutionHead
import copy
torch.manual_seed(114514)
torch.cuda.manual_seed(114514)
class Spec2sgn(nn.Module):
    """
    Masked Auto Encoder with Swin Transformer backbone
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, mask_ratio: float = 0.75, in_chans: int = 3,
                 decoder_embed_dim=512, norm_pix_loss=False,
                 depths: tuple = (2, 2, 6, 2), embed_dim: int = 96, num_heads: tuple = (3, 6, 12, 24),
                 window_size: int = 7, qkv_bias: bool = True, mlp_ratio: float = 4.,
                 drop_path_rate: float = 0.1, drop_rate: float = 0., attn_drop_rate: float = 0.,
                 norm_layer=None, patch_norm: bool = True, mask_type='mixmask', m=0.99,
                 attention_mask_is=False,channel_mean=True):
        super().__init__()
        self.mask_ratio = mask_ratio
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_size = patch_size
        self.mask_type = mask_type
        self.norm_pix_loss = norm_pix_loss
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.drop_path = drop_path_rate
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.norm_layer = norm_layer
        self.attention_mask_is=attention_mask_is
        self.channel_mean=channel_mean

        self.patch_embed = PatchEmbedding(patch_size=patch_size, in_c=in_chans, embed_dim=embed_dim,
                                          norm_layer=norm_layer if patch_norm else None)
        self.first_patch_expanding = PatchExpanding(dim=decoder_embed_dim, norm_layer=norm_layer)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim), requires_grad=False)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.layers = self.build_layers()
        self.layers_up = self.build_layers_up()
        self.norm_up = norm_layer(embed_dim)
        self.m = m
        self.IQ_pred = nn.Sequential(nn.Linear(embed_dim, 64),
                                     nn.Dropout(0.2),
                                     nn.Linear(64, 2))
        self.len_pred = nn.Sequential(nn.Linear(int(img_size/patch_size)**2, 256),
                                     nn.Dropout(0.2),
                                     nn.Linear(256, 128))
        self.attn_sgn = nn.Sequential(nn.Linear(decoder_embed_dim, decoder_embed_dim // 2),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop_rate),
                                      nn.Linear(decoder_embed_dim // 2, decoder_embed_dim),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop_rate),
                                      nn.Linear(decoder_embed_dim, decoder_embed_dim),
                                      nn.Sigmoid())
        self.initialize_weights()


    def initialize_weights(self):
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches ** .5), cls_token=False)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        torch.nn.init.normal_(self.mask_token, std=.02)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p ** 2 * 1)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], 1, h * p, h * p)
        return imgs

    def build_layers(self):
        layers = nn.ModuleList()
        for i in range(self.num_layers):
            layer = BasicBlock(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                norm_layer=self.norm_layer,
                attention_mask_is=self.attention_mask_is,
                patch_merging=False if i == self.num_layers - 1 else True)
            layers.append(layer)
        return layers
    def build_layers_up(self):
        layers_up = nn.ModuleList()
        for i in range(self.num_layers - 2):
            layer = BasicBlockUp(
                index=i,
                depths=self.depths,
                embed_dim=self.embed_dim,
                num_heads=self.num_heads,
                drop_path=self.drop_path,
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop_rate=self.drop_rate,
                attn_drop_rate=self.attn_drop_rate,
                patch_expanding=True if i < self.num_layers - 2 else False,
                norm_layer=self.norm_layer,
                decode_is=True)
            layers_up.append(layer)
        return layers_up

    def forward_encoder(self, clean):
        x_clean = self.patch_embed(clean)

        for i, layer in enumerate(self.layers):
            x_clean = layer(x_clean, x_clean)

        return  x_clean
    def forward_decoder(self, data):

        x = self.first_patch_expanding(data)

        for layer in self.layers_up:
            x = layer(x)

        return x

    def sgn_norm(self,sgn):
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_max, _ = torch.max(sgn_max, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min, _ = torch.min(sgn_min, dim=1)
        sgn_min=sgn_min.unsqueeze(1).unsqueeze(2)
        sgn_max=sgn_max.unsqueeze(1).unsqueeze(2)
        sgn = (2*sgn - sgn_min - sgn_max) / (sgn_max- sgn_min)
        return sgn

    def forward(self, clean):
        latent = self.forward_encoder(clean)
        latent = self.attn_sgn(latent) * latent
        latent = self.forward_decoder(latent)
        latent = latent.reshape(latent.shape[0], -1,latent.shape[3])
        latent = self.IQ_pred(latent)
        latent=rearrange(latent, 'B L C -> B C L')
        sgn=self.len_pred(latent)
        sgn = self.sgn_norm(sgn)
        freq=sgn[:,0,:]+1j*sgn[:,1,:]
        # freq=torch.abs(torch.fft.fft(freq))
        # freq_max,_=torch.max(freq,dim=1)
        # freq_min,_ = torch.min(freq, dim=1)
        # freq = (freq - freq_min.unsqueeze(1)) / (freq_max.unsqueeze(1) - freq_min.unsqueeze(1))
        # freq=torch.unsqueeze(freq,1)
        return sgn,freq

def filter_sgn(sgn, filiter='high', filiter_threshold=0.99, filiter_size=0.0, middle_zero=False, freq_smooth=False,
           return_IQ=False):
    I = sgn[0]
    Q = sgn[1]
    IQ = torch.view_as_complex(torch.stack((I, Q), dim=-1))  # 复数形式的IQ数据

    # 对复数数据进行傅里叶变换
    N = IQ.shape[0]
    IQ_fft = torch.fft.fft(IQ, n=N)
    IQ_abs = torch.abs(IQ_fft)
    sorted_indices = torch.argsort(IQ_abs)
    if filiter == 'high':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs >= threshold] *= filiter_size  # 将大于阈值的点设为0
    elif filiter == 'low':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs <= threshold] *= filiter_size  # 将大于阈值的点设为0
        if middle_zero:
            IQ_fft[20:110] = 0.001
        if freq_smooth:
            # 定义平滑窗口的大小
            window_size = 3

            # 创建一个新的数组，用于存储平滑后的数据
            smoothed_arr = torch.zeros_like(IQ_fft)

            # 对数组进行平滑处理
            for i in range(window_size, len(IQ_fft) - window_size):
                smoothed_arr[i] = torch.mean(IQ_fft[i - window_size:i + window_size])
    sgn_IQ = sgn.clone()
    sgn = torch.fft.ifft(IQ_fft)
    if return_IQ:
        sgn_IQ[0] = torch.view_as_real(sgn)[..., 0]
        sgn_IQ[1] = torch.view_as_real(sgn)[..., 1]
        return sgn_IQ
    else:
        return sgn

class Sgnc2freq(nn.Module):
    def __init__(self, d_input=128, d_model=256, d_ff=512, kernal_size=7, num_layers=7, num_heads=8, dilation=2,
                 attn_drop=0.2,emb_dropout=0.2):
        super(Sgnc2freq, self).__init__()
        self.CE = ConvolutionalEncoding(in_channels=2, temp_channels=d_input, out_channels=d_model,
                                        kernel_size=kernal_size)  # 卷积编码层
        self.freq_vector = nn.Parameter(torch.randn(d_model))  # 可学习的分类向量
        self.pos_embedding = nn.Parameter(torch.randn(1, d_input, d_model))

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                    dropout=attn_drop,dim_feedforward=d_ff)
        self.dropout = nn.Dropout(emb_dropout)
        # 使用编码器层创建一个Transformer编码器
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linears_D = nn.Sequential(
            nn.Linear(d_model, d_model // 2,bias=False),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 2,bias=False)
        )
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.attn_sgn = nn.Sequential(nn.Linear(d_model, d_model // 2,bias=False),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop),
                                      nn.Linear(d_model // 2, d_model,bias=False),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop),
                                      nn.Linear(d_model, d_model,bias=False),
                                      nn.Sigmoid())

        self.initialize_weights()
    def sgn_norm(self,sgn):
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_max, _ = torch.max(sgn_max, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min, _ = torch.min(sgn_min, dim=1)
        sgn_min=sgn_min.unsqueeze(1).unsqueeze(2)
        sgn_max=sgn_max.unsqueeze(1).unsqueeze(2)
        freq = (2*sgn - sgn_min - sgn_max) / (sgn_max- sgn_min)
        return freq

    def initialize_weights(self):

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):

            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        # 输入：(batch_size, in_channels, sequence_length)
        # 卷积编码
        # x_org = torch.clone(x)
        # for i in range(x.shape[0]):
        #     x[i] = filter_sgn(sgn=x[i], filiter='low', filiter_threshold=0.85, filiter_size=0.0,
        #                       middle_zero=True, freq_smooth=True, return_IQ=True)
        x = self.CE(x)
        x = x.permute(0, 2, 1)
        # 并入分类向量
        # x = torch.cat((self.freq_vector.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1), x), dim=1)
        # 位置编码
        x_ = x+self.pos_embedding
        x = self.dropout(x_)

        x = self.encoder(x)
        # 去掉类别向量
        # x = x[:, 1:, :]
        # 时间卷积头
        # x = self.TH(x)
        # x=x.permute(0, 2, 1)
        # x=self.linears_D(x)
        # x = x.permute(0, 2, 1)

        x = self.attn_sgn(x)*x+self.attn_sgn(x_)*x_
        x = self.dropout(x)
        # x=self.pool(x)
        sgn = self.linears_D(x)
        sgn = sgn.permute(0, 2, 1)
        sgn=self.sgn_norm(sgn)
        I=sgn[:,0,:]
        Q=sgn[:,1,:]
        freq = torch.abs(torch.fft.fft(I+Q*1j))

        return sgn,freq

class Sgnc2freq_fc(nn.Module):
    def __init__(self, d_input=128, d_model=256, d_ff=512, kernal_size=7, num_layers=7, num_heads=8, dilation=2,
                 attn_drop=0.2,num_classes=11):
        super(Sgnc2freq_fc, self).__init__()
        self.CE = ConvolutionalEncoding(in_channels=2, temp_channels=d_input, out_channels=d_model,
                                        kernel_size=kernal_size)  # 卷积编码层
        self.freq_vector = nn.Parameter(torch.randn(d_model))  # 可学习的分类向量
        self.PE = PositionalEncoding(d_model=d_model)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=num_heads,
                                    dropout=attn_drop,dim_feedforward=d_ff)

        # 使用编码器层创建一个Transformer编码器
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.linears_D = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Dropout(0.2),
            nn.Linear(d_model // 2, 2)
        )
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.attn_sgn = nn.Sequential(nn.Linear(d_model, d_model // 2),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop),
                                      nn.Linear(d_model // 2, d_model),
                                      nn.Tanh(),
                                      nn.Dropout(attn_drop),
                                      nn.Linear(d_model, d_model),
                                      nn.Sigmoid())

        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(d_model),
            nn.Linear(d_model, 32),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
    def sgn_norm(self,sgn):
        sgn_max, _ = torch.max(sgn, dim=1)
        sgn_max, _ = torch.max(sgn_max, dim=1)
        sgn_min, _ = torch.min(sgn, dim=1)
        sgn_min, _ = torch.min(sgn_min, dim=1)
        sgn_min=sgn_min.unsqueeze(1).unsqueeze(2)
        sgn_max=sgn_max.unsqueeze(1).unsqueeze(2)
        freq = (2*sgn - sgn_min - sgn_max) / (sgn_max- sgn_min)
        return freq

    def forward(self, x):
        x=torch.squeeze(x,1)
        x = self.CE(x)
        x = x.permute(0, 2, 1)


        x = self.encoder(x)

        x = self.attn_sgn(x)*x
        x = x.permute(0, 2, 1)
        x=self.pool(x)
        x = x.view(x.size(0), -1)
        x=self.mlp_head(x)

        return x

if __name__ == '__main__':
    import time
    img_size=128
    patch_size = 4
    window_size = 2
    in_channel=3
    model = Spec2sgn(
        img_size=128, patch_size=patch_size, in_chans=in_channel,
        decoder_embed_dim=384,
        depths=(2, 2, 2), embed_dim=96, num_heads=(3, 6, 12, 24),
        window_size=window_size, qkv_bias=True, mlp_ratio=4,
        drop_path_rate=0.2, drop_rate=0.2, attn_drop_rate=0.2,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), mask_ratio=0.5, mask_type='suiji',
        attention_mask_is=True, channel_mean=False)
    # model=Sgnc2freq(d_input=128,
    #                 d_model=256,
    #                 d_ff=512,
    #                 kernal_size=7,
    #                 num_layers=7,
    #                 num_heads=8,
    #                 dilation=2)
    noise = torch.randn((3, in_channel, img_size, img_size)).cuda()
    clean = torch.randn((3, in_channel, img_size, img_size)).cuda()
    sgn = torch.randn((3, 2, img_size)).cuda()
    model = model.cuda()
    Sgn=torch.randn((3, 2, img_size)).cuda()
    Freq = torch.randn((3, 1, img_size)).cuda()
    out=model(clean)
    print(out.shape)
    # sgn,freq = model(clean)
    # cretentation=nn.MSELoss()
    # loss1=cretentation(sgn,Sgn)
    # loss2=cretentation(freq,Freq)
    # loss=loss1+loss2
    # loss.backward()
