import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
import warnings
warnings.filterwarnings("ignore")
def DataTransform(sample,a=0.1,b=0.2,c=5,d=0.1):

    weak_aug = scaling(sample, a,b)

    strong_aug = frequency_shift(permutation(sample,c),d)

    return weak_aug, strong_aug

def frequency_shift(signal, shift=0.1):
    # 转换为numpy数组
    # 进行快速傅立叶变换 (FFT)
    signal_fft = np.fft.fft(signal)

    # 创建一个与信号形状相同的频率轴
    freq_axis = np.fft.fftfreq(signal.shape[-1])

    # 给频移的信号进行FFT逆变换，实部就是我们需要的时域信号
    signal_shifted = np.fft.ifft(signal_fft * np.exp(-1j * 2 * np.pi * shift * freq_axis))

    # 转回torch tensor
    signal_shifted = np.real(signal_shifted)

    return signal_shifted


def jitter(x, sigma=0.8):
    # https://arxiv.org/pdf/1706.00527.pdf

    return x + np.random.normal(loc=0., scale=sigma, size=x.shape)




def scaling(x, noise_level=0.1,rayle_leavel=0.2):
    # https://arxiv.org/pdf/1706.00527.pdf
    noise = np.random.normal(size=x.shape) * noise_level
    rayleigh_coeff = np.random.rayleigh(1, x.shape)*rayle_leavel
    faded_signal = x * (rayleigh_coeff+1) + noise
    # 生成瑞利分布的随机数，代表信道衰落系数

    return faded_signal


def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[:,warp]
        else:
            ret[i] = pat
    return ret


if __name__ == '__main__':
    import argparse
    import matplotlib.pyplot as plt
    from signeltoimage import pwvd
    import torch
    parser = argparse.ArgumentParser(description='Train TransNet')
    parser.add_argument("--noise_level", default=0.1)
    parser.add_argument("--max_seg", type=int, default=10)
    parser.add_argument("--shift", default=0.5)
    opt = parser.parse_args()
    args = parser.parse_args()
    data=np.load('../data_50/50classes_val_data.npy')
    for i in range(10):
        sgn=data[i]
        sgn1=np.expand_dims(sgn,0)
        # sgn1 = torch.tensor(sgn1)
        weak_aug, strong_aug=DataTransform(sgn1,a=0.2,b=0.2,c=10,d=0.5)

        '''
        a [0.1 0.5]
        '''
        # weak_aug=weak_aug.detach().cpu().numpy()
        # strong_aug = strong_aug.detach().cpu().numpy()
        strong_aug=np.squeeze(strong_aug,0)
        weak_aug = np.squeeze(weak_aug, 0)
        img_s = pwvd(strong_aug, norm='maxmin', resize_is=True, resize_num=128)
        img_w = pwvd(weak_aug, norm='maxmin', resize_is=True, resize_num=128)
        plt.pcolormesh(img_s[0])
        plt.pcolormesh(img_w[0])
        plt.show()
        # fig, axs = plt.subplots(2, 2)
        # axs[0,0].plot(weak_aug[0, :], label='I')
        # axs[0,0].plot(weak_aug[1,:], label='Q')
        # axs[0,1].plot(strong_aug[0, :], label='I')
        # axs[0,1].plot(strong_aug[1, :], label='Q')
        # axs[1, 0].plot(sgn[0, :], label='I')
        # axs[1, 0].plot(sgn[1, :], label='Q')
        # plt.show()