import torch
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import Block
from pwvdswin_ViT import CrossAttention_VIT
from torch import nn
def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches_noise : torch.Tensor,patches_clean : torch.Tensor):
        T, B, C = patches_noise.shape
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches_noise.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches_noise.device)

        patches_noise = take_indexes(patches_noise, forward_indexes)
        patches_noise = patches_noise[:remain_T]

        patches_clean = take_indexes(patches_clean, forward_indexes)
        patches_clean = patches_clean[:remain_T]

        return patches_noise,patches_clean, forward_indexes, backward_indexes

class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 in_channel=1,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 is_attn=True,
                 single=False,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        #注 Conv2d channel参数
        self.patchify = torch.nn.Conv2d(in_channel, emb_dim, patch_size, patch_size)
        self.is_attn=is_attn
        self.single=single
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,proj_drop=0.2,attn_drop=0.2) for _ in range(num_layer)])
        if self.is_attn:
            self.attn=CrossAttention_VIT(dim=emb_dim, L=int((image_size // patch_size)**2*(1-mask_ratio)+1),  attn_drop=0.2)
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, noise, clean):
        patches_noise = self.patchify(noise)
        patches_noise = rearrange(patches_noise, 'b c h w -> (h w) b c')
        patches_noise = patches_noise + self.pos_embedding

        patches_clean = self.patchify(clean)
        patches_clean = rearrange(patches_clean, 'b c h w -> (h w) b c')
        patches_clean = patches_clean + self.pos_embedding

        patches_noise,patches_clean, forward_indexes, backward_indexes = self.shuffle(patches_noise,patches_clean)
        if self.single==False:
            patches_noise = torch.cat([self.cls_token.expand(-1, patches_noise.shape[1], -1), patches_noise], dim=0)
            patches_noise = rearrange(patches_noise, 't b c -> b t c')
            features_noise = self.layer_norm(self.transformer(patches_noise))
            if self.is_attn:
                features_noise=self.attn(features_noise)
            features_noise = rearrange(features_noise, 'b t c -> t b c')

        patches_clean = torch.cat([self.cls_token.expand(-1, patches_clean.shape[1], -1), patches_clean], dim=0)
        patches_clean = rearrange(patches_clean, 't b c -> b t c')
        features_clean = self.layer_norm(self.transformer(patches_clean))
        if self.is_attn:
            features_clean = self.attn(features_clean)
        features_clean = rearrange(features_clean, 'b t c -> t b c')
        if self.single == False:
            return features_noise, features_clean, backward_indexes
        else:
            return features_clean, features_clean, backward_indexes

class MAE_Encoder_fc(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 in_channel=3,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 is_attn=True,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        #注 Conv2d channel参数
        self.patchify = torch.nn.Conv2d(in_channel, emb_dim, patch_size, patch_size)
        self.is_attn=is_attn
        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,proj_drop=0.2,attn_drop=0.2) for _ in range(num_layer)])
        if self.is_attn:
            self.attn=CrossAttention_VIT(dim=emb_dim, L=int((image_size // patch_size)**2+1),  attn_drop=0.2)
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, noise):
        patches_noise = self.patchify(noise)
        patches_noise = rearrange(patches_noise, 'b c h w -> (h w) b c')
        patches_noise = patches_noise + self.pos_embedding



        # patches_noise,patches_clean, forward_indexes, backward_indexes = self.shuffle(patches_noise,patches_noise)

        patches_noise = torch.cat([self.cls_token.expand(-1, patches_noise.shape[1], -1), patches_noise], dim=0)
        patches_noise = rearrange(patches_noise, 't b c -> b t c')
        features_noise = self.layer_norm(self.transformer(patches_noise))
        if self.is_attn:
            features_noise=self.attn(features_noise)

        return features_noise


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 in_channel=1,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head,proj_drop=0.2,attn_drop=0.2) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, patch_size ** 2*in_channel)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=image_size//patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:] # remove global feature

        patches = self.head(features)
        mask = torch.zeros_like(patches)
        mask[T:] = 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)
        mask = self.patch2img(mask)

        return img, mask

class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 in_channel=1,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size,in_channel, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.norm = torch.nn.BatchNorm1d(100)
        self.decoder = MAE_Decoder(image_size,in_channel, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, noise, clean):
        features_noise,features_clean, backward_indexes = self.encoder(noise, clean)
        predicted_img_noise, mask = self.decoder(features_noise,  backward_indexes)
        predicted_img_clean, _ = self.decoder(features_clean, backward_indexes)
        features_noise_out, features_clean_out=features_noise,features_clean
        features_noise_out=rearrange(features_noise_out, 't b c -> b c t')
        features_clean_out = rearrange(features_clean_out, 't b c -> b c t')
        features_noise_out = torch.mean(features_noise_out, -1)
        features_noise_out = features_noise_out.reshape(features_noise_out.shape[0], -1)
        features_clean_out = torch.mean(features_clean_out, -1)
        features_clean_out = features_clean_out.reshape(features_clean_out.shape[0], -1)
        
        return features_noise_out,features_clean_out,predicted_img_noise,predicted_img_clean,mask


class MAE_ViT_gasf(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 in_channel=1,
                 patch_size=2,
                 emb_dim=768,
                 decode_emb_dim=384,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, in_channel, patch_size, emb_dim,
                                   encoder_layer, encoder_head, mask_ratio,is_attn=False,single=True)
        self.norm = torch.nn.BatchNorm1d(100)
        if emb_dim!=decode_emb_dim:
            self.down_dim=nn.Linear(emb_dim,decode_emb_dim)
        else:
            self.down_dim = nn.Identity()
        self.decoder = MAE_Decoder(image_size, in_channel, patch_size, decode_emb_dim, decoder_layer, decoder_head)

    def forward(self,  clean):
        features_noise, features_clean, backward_indexes = self.encoder(clean, clean)
        features_clean=self.down_dim(features_clean)
        predicted_img_clean, mask = self.decoder(features_clean, backward_indexes)


        return predicted_img_clean, mask

class MAE_ViT_gasf_fc(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 in_channel=3,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 numclass=11
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder_fc(image_size, patch_size, emb_dim,in_channel,
                                      encoder_layer, encoder_head, mask_ratio,is_attn=False)
        self.mlp_head = nn.Sequential(nn.Linear(emb_dim, emb_dim*4),
                                      nn.GELU(),
                                      nn.Dropout(0.2),
                                      # nn.Linear(128, 64),
                                      # nn.ReLU(),
                                      # nn.Dropout(0.2),
                                      nn.Linear(emb_dim*4, numclass)
                                      )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, noise,clean):
        latent = self.encoder(noise)
        latent = self.avgpool(latent.transpose(1, 2))  # B C 1
        latent = torch.flatten(latent, 1)
        out = self.mlp_head(latent)

        return out

class MAE_ViT_fc2(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 numclass=11
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder_fc(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.linear_n = nn.Sequential(nn.Linear(emb_dim, 128),
                                      # nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(128, 64),
                                      # nn.ReLU(),
                                      nn.Dropout(0.2),
                                      nn.Linear(64, numclass)
                                      )
        self.avgpool = nn.AdaptiveAvgPool1d(1)
    def forward(self, noise,clean):
        latent = self.encoder(noise)
        latent = rearrange(latent, 'b t c -> t b c')
        out = self.linear_n(latent[0])
        return out


if __name__ == '__main__':
    # shuffle = PatchShuffle(0.75)
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a)
    # print(b.shape)

    img1 = torch.rand(2, 3, 128, 128)
    img2 = torch.rand(2, 3, 128, 128)
    model = MAE_ViT_gasf(image_size=128,
                         in_channel=3,
                         patch_size=8,
                         emb_dim=768,
                         decode_emb_dim=384,
                         encoder_layer=4,
                         encoder_head=8,
                         decoder_layer=2,
                         decoder_head=12,
                         mask_ratio=0.5,

                         )
    out=model(img1)
    # print(out.shape)
    '''
    features (17,2,256)->(2,256)
    '''
    # print(predicted_img_noise)

    def model_structure(model):
        blank = ' '
        print('-' * 90)
        print('|' + ' ' * 11 + 'weight name' + ' ' * 10 + '|' \
              + ' ' * 15 + 'weight shape' + ' ' * 15 + '|' \
              + ' ' * 3 + 'number' + ' ' * 3 + '|')
        print('-' * 90)
        num_para = 0
        type_size = 1  # 如果是浮点数就是4

        for index, (key, w_variable) in enumerate(model.named_parameters()):
            if len(key) <= 30:
                key = key + (30 - len(key)) * blank
            shape = str(w_variable.shape)
            if len(shape) <= 40:
                shape = shape + (40 - len(shape)) * blank
            each_para = 1
            for k in w_variable.shape:
                each_para *= k
            num_para += each_para
            str_num = str(each_para)
            if len(str_num) <= 10:
                str_num = str_num + (10 - len(str_num)) * blank

            print('| {} | {} | {} |'.format(key, shape, str_num))
        print('-' * 90)
        print('The total number of parameters: ' + str(num_para))
        print('The parameters of Model {}: {:4f}M'.format(model._get_name(), num_para * type_size / 1000 / 1000))
        print('-' * 90)

    model_structure(model)