import numpy as np
import torch
import random
import scipy.io as scio
from typing import Any, Callable, cast, Dict, List, Optional, Tuple
from torch.utils.data import Dataset  # Dataset是个抽象类，只能用于继承
from utils.signeltoimage import *
from utils.tezhengtiqu import *
import h5py
from utils.chazhi import *
def awgn(x, snr, seed=7):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''
    np.random.seed(seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / x.shape[1]
    npower = xpower / snr
    noise=np.zeros((x.shape[0],x.shape[1]))
    noise[0,:] = np.random.randn(x.shape[1]) * np.sqrt(npower)
    noise[1, :] = np.random.randn(x.shape[1]) * np.sqrt(npower)
    return x + noise

def l2_normalize(x, axis=-1):
    y = np.sum(x ** 2, axis, keepdims=True)
    return x / np.sqrt(y)

#dataset:RML2016.10a (train test 7:3)
class SigDataSet_stft(Dataset):
    def __init__(self, data_path,adsbis=False,newdata=False,snr_range=[1,7],resample_is=False,samplenum=15,norm='maxmin',chazhi=False,chazhinum=2,is_DAE=False):
        super().__init__()
        self.data = scio.loadmat(data_path)['data']
        self.labels = scio.loadmat(data_path)['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is=resample_is
        self.norm=norm
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum=samplenum
        self.is_DAE=is_DAE
        self.rml = True
        if (adsbis == False) and (newdata==False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata==True):
            self.rml=False
        self.newdata=newdata
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        self.sgn = self.data[item]
        # if self.is_DAE==True:
        #     self.sgn =l2_normalize(self.sgn)
        if self.chazhi == True:
            self.sgn = chpip(self.sgn, self.cnum)
        if self.rml == True:
            return torch.tensor(stp(self.sgn), dtype=torch.float32), torch.tensor(
                    self.sgn, dtype=torch.float32), torch.tensor(
                    np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                    self.labels[item], dtype=torch.long), torch.tensor(
                    self.snr[item], dtype=torch.long)
        elif self.adsbis == True:

            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR * 5 - 15


            self.sgn = awgn(self.sgn, self.SNR1)
            if self.resample_is==True:
                self.img=stp(resampe(self.sgn,samplenum=self.samplenum))
            else:
                self.img=stp(self.sgn,nperseg=100,noverlap=90)
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.sgn, dtype=torch.float32), torch.tensor(
                np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)
        elif self.newdata==True:
            self.sgn = toIQ(self.sgn)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR * 5 - 15
            self.sgn = awgn(self.sgn, self.SNR1)
            if self.is_DAE==True:
                self.sgn=l2_normalize(self.sgn)
            if self.resample_is == True:
                self.img = stp(resampe(self.sgn, samplenum=self.samplenum))
                self.sgn=resampe(self.sgn, samplenum=self.samplenum)
            else:
                self.img=stp(self.sgn)
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.sgn, dtype=torch.float32), torch.tensor(
                np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)



class SigDataSet_pwvd(Dataset):
    def __init__(self, data_path,adsbis=False,newdata=False,snr_range=[1,7],resample_is=False,samplenum=15,norm='no'
                 ,chazhi=False,chazhinum=2,is_DAE=False,resize_is=False):
        super().__init__()
        self.data = scio.loadmat(data_path)['data']
        self.labels = scio.loadmat(data_path)['label'].flatten()
        self.snrmin = snr_range[0]
        self.snrmax = snr_range[1]
        self.adsbis = adsbis
        self.resample_is=resample_is
        self.norm=norm
        self.resize_is=resize_is
        self.chazhi = chazhi
        self.cnum = chazhinum
        self.samplenum=samplenum
        self.is_DAE=is_DAE
        self.rml = True
        if (adsbis == False) and (newdata==False):
            self.snr = scio.loadmat(data_path)['snr'].flatten()
        if (adsbis == True) or (newdata==True):
            self.rml=False
        self.newdata=newdata
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        self.sgn = self.data[item]
        if self.is_DAE==True:
            self.sgn =l2_normalize(self.sgn)
        if self.chazhi == True:
            self.sgn = chpip(self.sgn, self.cnum)
        if self.rml == True:
            return torch.tensor(pwvd(self.sgn, self.norm,self.resize_is), dtype=torch.float32), torch.tensor(
                    self.sgn, dtype=torch.float32), torch.tensor(
                    np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                    self.labels[item], dtype=torch.long), torch.tensor(
                    self.snr[item], dtype=torch.long)
        elif self.adsbis == True:

            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR * 5 - 15


            self.sgn = awgn(self.sgn, self.SNR1)
            if self.resample_is==True:
                self.img=pwvd(resampe(self.sgn,samplenum=self.samplenum), self.norm,self.resize_is)
                # self.img =self.sgn
                self.sgn = resampe(self.sgn, samplenum=self.samplenum)
            else:
                # self.img=pwvd(self.sgn)
                self.img = self.sgn
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.sgn, dtype=torch.float32), torch.tensor(
                np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)
        elif self.newdata==True:
            self.sgn = toIQ(self.sgn)
            self.SNR = random.randint(self.snrmin, self.snrmax)
            self.SNR1 = self.SNR * 5 - 15
            self.sgn = awgn(self.sgn, self.SNR1)
            if self.is_DAE==True:
                self.sgn=l2_normalize(self.sgn)
            if self.resample_is == True:
                self.img = pwvd(resampe(self.sgn, samplenum=self.samplenum))
                self.sgn=resampe(self.sgn, samplenum=self.samplenum)
            else:
                self.img = pwvd(self.sgn)
            return torch.tensor(self.img, dtype=torch.float32), torch.tensor(
                self.sgn, dtype=torch.float32), torch.tensor(
                np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
                self.labels[item], dtype=torch.long), torch.tensor(
                self.SNR, dtype=torch.long)


class SigDataSet_spwvd(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data = scio.loadmat(data_path)['data']
        self.labels = scio.loadmat(data_path)['label'].flatten()
        self.snr=scio.loadmat(data_path)['snr'].flatten()


    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        self.sgn = self.data[item]

        return torch.tensor(spwvd(self.sgn), dtype=torch.float32), torch.tensor(
            self.sgn, dtype=torch.float32), torch.tensor(
            np.expand_dims(tezheng(self.sgn), axis=0), dtype=torch.float32), torch.tensor(
            self.labels[item],dtype=torch.long),torch.tensor(
            self.snr[item],dtype=torch.long)


