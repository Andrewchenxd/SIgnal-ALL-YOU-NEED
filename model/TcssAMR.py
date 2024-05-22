# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn
import numpy as np
import copy
from math import sqrt
import torch
from torch import nn
import torch.nn.functional as F
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
            # nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.,num_patches=128):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)
        self.pos_embedding = nn.Parameter(torch.randn(1, heads , num_patches,num_patches))
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots+=self.pos_embedding
        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, attn_mask):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        return context.transpose(2, 1).contiguous()

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads,
                 d_keys=None, d_values=None, mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model//n_heads)
        d_values = d_values or (d_model//n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries):
        keys, values, attn_mask = queries,queries,None
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out= self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout,num_patche ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout,
                            num_patches=num_patche)),
                # AttentionLayer(ProbAttention(False, factor=5, attention_dropout=dropout, output_attention=False),
                #                    d_model=dim, n_heads= heads, mix=True),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class conv_patch_embedding(nn.Module):
    def __init__(self,in_channel=1,out_channel=80):
        super(conv_patch_embedding, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channel,256,(1,3),padding=(0,1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256,256,(2,3),padding=(0,1)),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(256, out_channel, (1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    def forward(self,x):
        out = self.encoder(x)
        out = torch.squeeze(out,dim=2)
        out=torch.transpose(out,1,2)
        return out

class Attention_pool(nn.Module):
    def __init__(self,sgn_len=128):
        super(Attention_pool, self).__init__()
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.semodel = nn.Sequential(
            nn.Linear(sgn_len,sgn_len//8),
            nn.ReLU(),
            nn.Linear(sgn_len//8, sgn_len),
            nn.Sigmoid(),
        )
    def forward(self,x):
        out = self.pool(x)
        out = torch.squeeze(out, dim=2)
        out=self.semodel(out)
        out = torch.unsqueeze(out, dim=1)
        out = torch.matmul(out, x)
        out=torch.squeeze(out, dim=1)
        return out

class ProjectionHead(nn.Module):
    def __init__(self,in_shape,out_shape=256):
        super().__init__()
        hidden_shape = out_shape//2

        self.layer=nn.Sequential(
            nn.Linear(in_shape, hidden_shape),
            nn.BatchNorm1d(hidden_shape),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_shape, out_shape),
        )

    def forward(self,x):
        x = self.layer(x)
        return x
class ViT_TcssAMR(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls',
                 channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.,proj=False,proj_dim=256,nmb_prototypes=30):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = conv_patch_embedding(in_channel=1,out_channel=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, image_size , dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout,image_size)

        self.pool = Attention_pool(image_size)
        self.proj=proj
        if self.proj==False:
            self.to_latent = nn.Identity()
        else:
            self.projhead_dim = proj_dim
            self.nmb_prototypes = nmb_prototypes
            self.to_latent =ProjectionHead(dim,proj_dim)



    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)

        # x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)
        # x=x[:,:n]
        x = self.pool(x)
        if self.proj==True:

            x = self.to_latent(x)
            x = nn.functional.normalize(x, dim=1, p=2)
        else:
            x = self.to_latent(x)
        return x

class ViT_TcssAMR_fc(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = 2
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = conv_patch_embedding(in_channel=1,out_channel=dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout, image_size)

        self.pool = Attention_pool(image_size)

        self.mlp_head = nn.Sequential(
            nn.BatchNorm1d(dim),
            nn.Linear(dim, 32),
            # nn.ReLU(),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        # cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        # x = torch.cat((cls_tokens, x), dim=1)
        # pos = self.pos_embedding[:, :(n)]
        # x += pos
        x = self.dropout(x)

        x = self.transformer(x)
        # x=x[:,:n]
        x = self.pool(x)
        x=self.mlp_head(x)


        return x

class TcssAMR(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(TcssAMR, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = copy.deepcopy(base_encoder)
        self.encoder_k = copy.deepcopy(base_encoder)
        mlp = False


        if mlp:  # hack: brute-force replacement
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_q.fc)
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), self.encoder_k.fc)


        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient


    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)



    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long)).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2):
        '''
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1, q2, k1, k2
        '''

        q1 = self.encoder_q(x1)

        # q1 = self.predictor(q1)
        # q2 = self.predictor(q2)


        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            k2 = self.encoder_k(x2)  # keys: NxC
        loss=self.contrastive_loss(q1, k2)

        return  loss

class SWAV_TcssAMR(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07,epsilon=0.05,sinkhorn_iterations=10):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(SWAV_TcssAMR, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.epsilon = epsilon
        self.sinkhorn_iterations = sinkhorn_iterations
        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = copy.deepcopy(base_encoder)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.prototypes = nn.Linear(base_encoder.projhead_dim, base_encoder.nmb_prototypes, bias=False)


    @torch.no_grad()
    def distributed_sinkhorn(self,out,world_size=1):
        Q = torch.exp(out / self.epsilon).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()

    def contrastive_loss(self, scores_q, scores_k):
        temperature = self.T
        q_q=self.distributed_sinkhorn(scores_q)
        q_k = self.distributed_sinkhorn(scores_k)
        p_t=F.log_softmax(scores_q/temperature, dim=1)
        p_k = F.log_softmax(scores_k / temperature, dim=1)
        loss=-torch.mean(torch.sum(q_q*p_k+q_k*p_t, dim=1))/2
        # loss = -torch.mean(torch.sum(q_k * p_t, dim=1))
        return loss

    def forward(self, x1, x2):
        '''
        Input:
            x1: a batch of query images
            x2: a batch of key images
        Output:
            q1, q2, k1, k2
        '''
        code_q = self.encoder_q(x1)

        code_k = self.encoder_q(x2)  # keys: NxC
        # compute key features
        B = code_q.shape[0]
        score = torch.cat((code_q, code_k), 0)
        score = self.prototypes(score)
        score_q, score_k = torch.split(score, B, 0)
        loss = self.contrastive_loss(score_q, score_k)

        return  loss

class TcssAMR_fc(nn.Module):
    def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=False):

        super(TcssAMR_fc, self).__init__()


        self.encoder_q = copy.deepcopy(base_encoder)


    def forward(self, x1):

        out = self.encoder_q(x1)


        return out

if __name__ == '__main__':

    img_size=128
    channels=1
    moco_t=0.07
    embedding_dim=80

    v = ViT_TcssAMR(
        image_size=img_size,
        channels = channels,
        patch_size=1,
        num_classes=10,
        dim=embedding_dim,
        depth=2,
        heads=8,
        mlp_dim=160,
        dropout=0.1,
        emb_dropout=0.1,
        proj_dim=256,
        proj=True,
        nmb_prototypes=30
    )
    model = SWAV_TcssAMR(
        v,
        embedding_dim, 65536, 0.99, 0.07,epsilon=0.05,sinkhorn_iterations=3)

    model=model.cuda()
    sgn1= torch.randn((2, channels,2, img_size)).cuda()
    sgn2 = torch.randn((2, channels,2, img_size)).cuda()
    loss = model(sgn1,sgn2)
    print(loss)
    # model_fc(images2)
    def count_parameters_in_MB(model):
        return sum(p.numel() for p in model.parameters()) * 4 / (1024 ** 2)


    num_params = count_parameters_in_MB(v)
    print(f'Number of parameters: {num_params}')