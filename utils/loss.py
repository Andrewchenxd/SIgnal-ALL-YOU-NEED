import numpy as np
import torch
from torch.nn import Module
import torch.nn.functional as F

class SoftBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * p) * log(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
        as_pseudo_label (bool): Stop gradient propagation for the term ``(1 - beta) * p``.
            Can be interpreted as pseudo-label.
    """
    def __init__(self, beta=0.95, reduce=True, as_pseudo_label=True):
        super(SoftBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce
        self.as_pseudo_label = as_pseudo_label

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        y_pred_a = y_pred.detach() if self.as_pseudo_label else y_pred
        # second term = - (1 - beta) * p * log(p)
        bootstrap = - (1.0 - self.beta) * torch.sum(F.softmax(y_pred_a, dim=1) * F.log_softmax(y_pred, dim=1), dim=1)

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap


class HardBootstrappingLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, beta=0.8, reduce=True):
        super(HardBootstrappingLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y):
        # cross_entropy = - t * log(p)
        beta_xentropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        # z = argmax(p)
        z = F.softmax(y_pred.detach(), dim=1).argmax(dim=1)
        z = z.view(-1, 1)
        bootstrap = F.log_softmax(y_pred, dim=1).gather(1, z).view(-1)
        # second term = (1 - beta) * z * log(p)
        bootstrap = - (1.0 - self.beta) * bootstrap

        if self.reduce:
            return torch.mean(beta_xentropy + bootstrap)
        return beta_xentropy + bootstrap

class TripletLoss(Module):
    def __init__(self, margin=0.3):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)  # 获得一个简单的距离triplet函数

    def forward(self, inputs, labels):

        n = inputs.size(0)  # 获取batch_size，这里的inputs就是输入矩阵,即batchsize * 特征维度
        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)  # 每个数平方后， 进行加和（通过keepdim保持2维），再扩展成nxn维
        dist = dist + dist.t()  # 这样每个dis[i][j]代表的是第i个特征与第j个特征的平方的和
        dist.addmm_(1, -2,inputs, inputs.t())  # 然后减去2倍的 第i个特征*第j个特征 从而通过完全平方式得到 (a-b)^2
        dist = dist.clamp(min=1e-12).sqrt()  # 然后开方

        # For each anchor, find the hardest positive and negative
        mask = labels.expand(n, n).eq(labels.expand(n, n).t())  # 这里mask[i][j] = 1代表i和j的label相同， =0代表i和j的label不相同
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))  # 在i与所有有相同label的j的距离中找一个最大的
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))  # 在i与所有不同label的j的距离找一个最小的
        dist_ap = torch.cat(dist_ap)  # 将list里的tensor拼接成新的tensor
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge loss
        y = torch.ones_like(dist_an)  # 声明一个与dist_an相同shape的全1tensor
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss

class CenterLoss(Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=90, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss

class snrLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, beta=0.3, reduce=True):
        super(snrLoss, self).__init__()
        self.beta = beta
        self.reduce = reduce

    def forward(self, y_pred, y,snr):
        # cross_entropy = - t * log(p)
        beta_entropy = self.beta * F.cross_entropy(y_pred, y, reduction='none')

        # z = argmax(p)
        snr=np.where(snr>=10,1,0)
        snr=torch.tensor(snr,device=y.device)
        snr_entropy =  (1.0 - self.beta) * F.cross_entropy(y_pred, y, reduction='none')*snr

        if self.reduce:
            return torch.mean(beta_entropy + snr_entropy)
        return beta_entropy + snr_entropy

import torch.nn as nn
class selfsuprLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, temp=0.3, reduce=True):
        super(selfsuprLoss, self).__init__()
        self.temp = temp
        self.reduce = reduce

    def forward(self, output_label,q):
        # cross_entropy = - t * log(p)
        logsoftmax = nn.LogSoftmax(dim=1)
        output_label=logsoftmax(output_label/self.temp)
        loss=-q*output_label


        if self.reduce:
            return torch.mean(loss)
        return loss
class mocoLoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self, temp=0.3, reduce=True):
        super(mocoLoss, self).__init__()
        self.temp = temp
        self.reduce = reduce

    def forward(self, ql,list):
        # cross_entropy = - t * log(p)
        d=len(list)-1
        qkp=torch.exp(ql*list[d]/self.temp)
        if d==0:
            qkn =qkp
        else:
            for i in range(d):
                if i==0:
                    qkn=torch.exp(ql*list[i]/self.temp)
                else:
                    qkn += torch.exp(ql * list[i] / self.temp)

        loss = -torch.log(qkp/qkn)

        if self.reduce:
            return torch.mean(loss)
        return loss

class MAELoss(Module):
    """
    ``Loss(t, p) = - (beta * t + (1 - beta) * z) * log(p)``
    where ``z = argmax(p)``
    Args:
        beta (float): bootstrap parameter. Default, 0.95
        reduce (bool): computes mean of the loss. Default, True.
    """
    def __init__(self,  is_mae=True,norm_pix_loss=False,beta=0.1,no_mask=False,is_hec=False,vit_simple=False):
        super(MAELoss, self).__init__()

        self.is_mae = is_mae
        self.is_hec = is_hec
        self.beta=beta
        self.no_mask=no_mask
        self.vit_simple=vit_simple
        self.norm_pix_loss=norm_pix_loss
    def forward(self, imgs, pred, mask):
        if self.is_hec==False:
            if self.norm_pix_loss:
                mean = imgs.mean(dim=-1, keepdim=True)
                var = imgs.var(dim=-1, keepdim=True)
                imgs = (imgs - mean) / (var + 1.e-6) ** .5
            if self.is_mae=='rmse':
                loss = (pred - imgs)**2
                # loss = torch.abs(pred - imgs)
                loss = loss.mean(dim=-1)
                loss=torch.sqrt(loss)
            elif self.is_mae=='mae':
                # loss = (pred - imgs) ** 2
                loss = torch.abs(pred - imgs)
                loss = loss.mean(dim=-1)
                # loss = torch.sqrt(loss)
            elif self.is_mae == 'smoothl1':

                lossf=nn.SmoothL1Loss(reduction='none', beta=self.beta)
                loss = lossf(pred,imgs)
                loss = loss.mean(dim=-1)
            elif self.is_mae == 'mse':

                loss = (pred - imgs) ** 2
                # loss = torch.abs(pred - imgs)
                # loss = loss.mean(dim=-1)

            if self.no_mask==False:
                '''
                mask里1是掩码，0是未掩码
                '''
                loss = (loss * mask).sum() / mask.sum()
            else:
                loss = loss.mean()

            return loss
        else:
            losstotal=0
            losslast=0
            for i in range(len(pred)):
                pred_temp=pred[i]
                if self.norm_pix_loss:
                    mean = imgs.mean(dim=-1, keepdim=True)
                    var = imgs.var(dim=-1, keepdim=True)
                    imgs = (imgs - mean) / (var + 1.e-6) ** .5
                if self.is_mae == 'rmse':
                    loss = (pred_temp - imgs) ** 2
                    # loss = torch.abs(pred_temp - imgs)
                    loss = loss.mean(dim=-1)
                    loss = torch.sqrt(loss)
                elif self.is_mae == 'mae':
                    # loss = (pred_temp - imgs) ** 2
                    loss = torch.abs(pred_temp - imgs)
                    loss = loss.mean(dim=-1)
                    # loss = torch.sqrt(loss)
                elif self.is_mae == 'smoothl1':

                    lossf = nn.SmoothL1Loss(reduction='none', beta=self.beta)
                    loss = lossf(pred_temp, imgs)
                    loss = loss.mean(dim=-1)
                elif self.is_mae == 'mse':

                    loss = (pred_temp - imgs) ** 2
                    # loss = torch.abs(pred_temp - imgs)
                    loss = loss.mean(dim=-1)

                if self.no_mask == False:
                    loss = (loss * mask).sum() / mask.sum()
                else:
                    loss = loss.mean()
                if i==(len(pred)-1):
                    losslast=loss
                losstotal=losstotal+loss


            return losstotal/len(pred),losslast

class maeLoss(torch.nn.Module):
  """
  MAE 损失函数

  Args:
    reduction: 'mean' | 'sum' | 'none'
  """

  def __init__(self, reduction='mean'):
    super().__init__()
    self.reduction = reduction

  def forward(self, y_pred, y_true):
    """
    计算 MAE 损失

    Args:
      y_pred: 预测值
      y_true: 真实值

    Returns:
      MAE 损失值
    """

    loss = torch.abs(y_pred - y_true)

    if self.reduction == 'mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)

    return loss

class CosineSimilarityLoss(torch.nn.Module):

  def __init__(self, reduction='mean',dim=1 ,eps= 1e-8 ):
    super().__init__()
    self.reduction = reduction
    self.dim = dim
    self.eps = eps
  def forward(self, y_pred, y_true):


    loss = 1-F.cosine_similarity(y_pred, y_true, self.dim, self.eps)

    if self.reduction == 'mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)

    return loss

class InfoNCE(nn.Module):
    """
    Calculates the InfoNCE loss for self-supervised learning.
    This contrastive loss enforces the embeddings of similar (positive) samples to be close
        and those of different (negative) samples to be distant.
    A query embedding is compared with one positive key and with one or more negative keys.
    References:
        https://arxiv.org/abs/1807.03748v2
        https://arxiv.org/abs/2010.05113
    Args:
        temperature: Logits are divided by temperature before calculating the cross entropy.
        reduction: Reduction method applied to the output.
            Value must be one of ['none', 'sum', 'mean'].
            See torch.nn.functional.cross_entropy for more details about each option.
        negative_mode: Determines how the (optional) negative_keys are handled.
            Value must be one of ['paired', 'unpaired'].
            If 'paired', then each query sample is paired with a number of negative keys.
            Comparable to a triplet loss, but with multiple negatives per sample.
            If 'unpaired', then the set of negative keys are all unrelated to any positive key.
    Input shape:
        query: (N, D) Tensor with query samples (e.g. embeddings of the input).
        positive_key: (N, D) Tensor with positive samples (e.g. embeddings of augmented input).
        negative_keys (optional): Tensor with negative samples (e.g. embeddings of other inputs)
            If negative_mode = 'paired', then negative_keys is a (N, M, D) Tensor.
            If negative_mode = 'unpaired', then negative_keys is a (M, D) Tensor.
            If None, then the negative keys for a sample are the positive keys for the other samples.
    Returns:
         Value of the InfoNCE Loss.
     Examples:
        >>> loss = InfoNCE()
        >>> batch_size, num_negative, embedding_size = 32, 48, 128
        >>> query = torch.randn(batch_size, embedding_size)
        >>> positive_key = torch.randn(batch_size, embedding_size)
        >>> negative_keys = torch.randn(num_negative, embedding_size)
        >>> output = loss(query, positive_key, negative_keys)
    """

    def __init__(self, temperature=0.1, reduction='mean', negative_mode='unpaired',latent_all=False):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
        self.negative_mode = negative_mode
        self.latent_all=latent_all

    def forward(self, query, positive_key, negative_keys=None):
        if self.latent_all==False:
            return info_nce(query, positive_key, negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)
        else:
            loss=0
            for i in range(len(query)):
                loss=loss+info_nce(query[i], positive_key[i], negative_keys,
                        temperature=self.temperature,
                        reduction=self.reduction,
                        negative_mode=self.negative_mode)
            return loss/len(query)

def info_nce(query, positive_key, negative_keys=None, temperature=0.1, reduction='mean', negative_mode='unpaired'):
    # Check input dimensionality.
    if query.dim() != 2:
        raise ValueError('<query> must have 2 dimensions.')
    if positive_key.dim() != 2:
        raise ValueError('<positive_key> must have 2 dimensions.')
    if negative_keys is not None:
        if negative_mode == 'unpaired' and negative_keys.dim() != 2:
            raise ValueError("<negative_keys> must have 2 dimensions if <negative_mode> == 'unpaired'.")
        if negative_mode == 'paired' and negative_keys.dim() != 3:
            raise ValueError("<negative_keys> must have 3 dimensions if <negative_mode> == 'paired'.")

    # Check matching number of samples.
    if len(query) != len(positive_key):
        raise ValueError('<query> and <positive_key> must must have the same number of samples.')
    if negative_keys is not None:
        if negative_mode == 'paired' and len(query) != len(negative_keys):
            raise ValueError("If negative_mode == 'paired', then <negative_keys> must have the same number of samples as <query>.")

    # Embedding vectors should have same number of components.
    if query.shape[-1] != positive_key.shape[-1]:
        raise ValueError('Vectors of <query> and <positive_key> should have the same number of components.')
    if negative_keys is not None:
        if query.shape[-1] != negative_keys.shape[-1]:
            raise ValueError('Vectors of <query> and <negative_keys> should have the same number of components.')

    # Normalize to unit vectors
    query, positive_key, negative_keys = normalize(query, positive_key, negative_keys)
    if negative_keys is not None:
        # Explicit negative keys

        # Cosine between positive pairs
        positive_logit = torch.sum(query * positive_key, dim=1, keepdim=True)

        if negative_mode == 'unpaired':
            # Cosine between all query-negative combinations
            negative_logits = query @ transpose(negative_keys)

        elif negative_mode == 'paired':
            query = query.unsqueeze(1)
            negative_logits = query @ transpose(negative_keys)
            negative_logits = negative_logits.squeeze(1)

        # First index in last dimension are the positive samples
        logits = torch.cat([positive_logit, negative_logits], dim=1)
        labels = torch.zeros(len(logits), dtype=torch.long, device=query.device)
    else:
        # Negative keys are implicitly off-diagonal positive keys.

        # Cosine between all combinations
        logits = query @ transpose(positive_key)

        # Positive keys are the entries on the diagonal
        labels = torch.arange(len(query), device=query.device)

    return F.cross_entropy(logits / temperature, labels, reduction=reduction)

def transpose(x):
    return x.transpose(-2, -1)

def normalize(*xs):
    return [None if x is None else F.normalize(x, dim=-1) for x in xs]


class LogSpectralDistanceLoss(torch.nn.Module):
    def __init__(self, n_fft=128, hop_length=3, win_length=5):
        super(LogSpectralDistanceLoss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, output, target):
        """
        计算Log Spectral Distance损失
        Args:
            output (Tensor): 预测信号, 形状为(B, 2, L)
            target (Tensor): 目标信号, 形状为(B, 2, L)
        Returns:
            Tensor: Log Spectral Distance损失, 形状为scalar
        """
        I = output[:, 0, :]
        Q = output[:, 1, :]
        It = target[:, 0, :]
        Qt = target[:, 1, :]
        output = I + Q * 1j
        target = It + Qt * 1j
        # 计算短时傅里叶变换
        output_spec = torch.stft(output, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                 return_complex=False)
        target_spec = torch.stft(target, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                                 return_complex=False)

        # 计算功率谱
        output_power_spec = output_spec.pow(2).sum(-1)
        target_power_spec = target_spec.pow(2).sum(-1)

        # 计算对数谱
        output_log_spec = torch.log(torch.clamp(output_power_spec, min=1e-8))
        target_log_spec = torch.log(torch.clamp(target_power_spec, min=1e-8))

        # 计算对数谱差异
        log_spec_diff = output_log_spec - target_log_spec

        # 计算平方误差
        squared_error = log_spec_diff.pow(2)

        # 在频率维度上求和
        summed_squared_error = squared_error.sum(-1)

        # 计算平均平方误差
        n_freq = output_spec.size(-2)
        averaged_squared_error = summed_squared_error / n_freq

        # 计算LSD损失
        lsd_loss = torch.sqrt(2 * averaged_squared_error)

        return lsd_loss.mean()