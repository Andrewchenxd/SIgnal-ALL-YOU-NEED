import torch.nn as nn
import torch
import torch.nn.functional as F
import math


class TemporalConvolutionHead(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout_prob=0.1):
        super(TemporalConvolutionHead, self).__init__()
        self.casualpad = nn.ZeroPad2d(padding=((kernel_size - 1) * dilation, 0, 0, 0))
        self.dilated_conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.dilated_conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x):
        # 输入：(batch_size, in_channels, sequence_length)
        out = self.dropout1(self.relu(self.bn1(self.dilated_conv1(self.casualpad(x)))))
        out = self.dropout2(self.relu(self.bn2(self.dilated_conv2(self.casualpad(out)))))
        # 跨层连接
        out += x
        return out


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff):
        super(PoswiseFeedForwardNet, self).__init__()
        self.MLP = nn.Sequential(
            nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1),
            nn.Dropout(0.1)
        )
        # self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        # self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, len_q, d_model]
        output = self.MLP(inputs.transpose(1, 2))
        output = self.dropout(output).transpose(1, 2)
        return output + residual


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = int(d_model / num_heads)
        self.v_dim = self.k_dim

        # 定义线性投影层，用于将输入变换到多头注意力空间
        self.proj_q = nn.Linear(d_model, self.k_dim * num_heads, bias=False)
        self.proj_k = nn.Linear(d_model, self.k_dim * num_heads, bias=False)
        self.proj_v = nn.Linear(d_model, self.v_dim * num_heads, bias=False)
        # 定义多头注意力的线性输出层
        self.proj_o = nn.Linear(self.v_dim * num_heads, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, d_model = x.size()
        # 对输入进行线性投影, 将每个头的查询、键、值进行切分和拼接
        q = self.proj_q(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k = self.proj_k(x).view(batch_size, seq_len, self.num_heads, self.k_dim).permute(0, 2, 3, 1)
        v = self.proj_v(x).view(batch_size, seq_len, self.num_heads, self.v_dim).permute(0, 2, 1, 3)
        # 计算注意力权重和输出结果
        attn = torch.matmul(q, k) / self.k_dim ** 0.5  # 注意力得分

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)  # 注意力权重参数
        output = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len, -1)  # 输出结果
        # 对多头注意力输出进行线性变换和输出
        output = self.proj_o(output)

        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads):
        super(EncoderLayer, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.enc_self_attn = MultiHeadAttention(d_model, n_heads)
        self.pos_ffn = PoswiseFeedForwardNet(d_model, d_ff)
        self.dropout = nn.Dropout(0.1)

    def forward(self, enc_inputs):
        ## 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model] 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs = self.enc_self_attn(enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.dropout(enc_outputs) + enc_inputs
        # enc_outputs = self.pos_ffn(enc_outputs) # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, num_layers, num_heads):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_layers)])

    def forward(self, enc_outputs):
        for layer in self.layers:
            enc_outputs = layer(enc_outputs)
        return enc_outputs


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 加入位置编码
        x = x + self.pe[:, :x.size(1), :]
        return x


class ConvolutionalEncoding(nn.Module):
    def __init__(self, in_channels, temp_channels, out_channels, kernel_size):
        super(ConvolutionalEncoding, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, temp_channels, kernel_size,padding=int(kernel_size-1)//2)  # , padding='same'
        self.bn1 = nn.BatchNorm1d(temp_channels)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv1d(temp_channels, out_channels, kernel_size,padding=int(kernel_size-1)//2)  # , padding='same'
        self.bn2 = nn.BatchNorm1d(out_channels)


        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size//2,
                               padding=int(kernel_size//2 - 1) // 2)  # , padding='same'
        self.bn3 = nn.BatchNorm1d(out_channels)


    def forward(self, x):
        # 输入：(batch_size, in_channels, sequence_length)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out=out+self.relu(self.bn3(self.conv3(x)))
        return out


class SignalTransformer(nn.Module):
    def __init__(self, d_input=128, d_model=256, d_ff=512, kernal_size=7, num_layers=7, num_heads=8, dilation=2,
                 classes=11):
        super(SignalTransformer, self).__init__()
        self.CE = ConvolutionalEncoding(in_channels=2, temp_channels=d_input, out_channels=d_model,
                                        kernel_size=kernal_size)  # 卷积编码层
        self.classification_vector = nn.Parameter(torch.randn(d_model))  # 可学习的分类向量
        self.PE = PositionalEncoding(d_model=d_model)
        self.encoder = TransformerEncoder(d_model, d_ff, num_layers, num_heads)
        self.TH = TemporalConvolutionHead(in_channels=1, out_channels=1, kernel_size=kernal_size, dilation=dilation)
        self.linears = nn.Sequential(
            nn.Linear(d_model, d_input),
            nn.Linear(d_input, d_input),
            nn.Linear(d_input, classes)
        )



    def forward(self, x):
        # 输入：(batch_size, in_channels, sequence_length)
        # 卷积编码

        x = self.CE(x)
        x = x.permute(0, 2, 1)
        # 并入分类向量
        x = torch.cat((self.classification_vector.unsqueeze(0).expand(x.size(0), -1).unsqueeze(1), x), dim=1)
        # 位置编码
        x = self.PE(x)

        x = self.encoder(x)
        # 提取分类向量
        x = x[:, 0:1, :]
        # 时间卷积头
        x = self.TH(x)
        x = self.linears(x[:, 0, :])
        return x


if __name__ == "__main__":
    SiT = SignalTransformer(d_input=128,
                            d_model=256,
                            d_ff=512,
                            kernal_size=7,
                            num_layers=7,
                            num_heads=8,
                            dilation=2,
                            classes=11)
    input_data = torch.randn(64, 2, 128)
    x = SiT(input_data)
    # print(SiT)

