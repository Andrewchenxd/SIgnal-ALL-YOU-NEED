import pywt
import numpy as np
from scipy.signal import resample,hilbert2

def toIQ(sgn):
    newsgn = np.zeros((2, sgn.shape[0]))
    # y = hilbert2(sgn)
    newsgn[0] = np.real(sgn)
    newsgn[1] = np.imag(sgn)
    return newsgn

def Mult_Wave_denoise(noisy_data):
    noisy_data=noisy_data[0]+noisy_data[1]* 1j
    wavelet = pywt.Wavelet('db8')  # 使用Daubechies8小波
    levels = pywt.dwt_max_level(data_len=noisy_data.size, filter_len=wavelet.dec_len)

    # 对实部和虚部分别进行小波分解和阈值处理
    denoised_data = np.empty_like(noisy_data)
    for i,part in enumerate([noisy_data.real, noisy_data.imag]):
        # 执行多级小波分解
        coeffs = pywt.wavedec(part, wavelet, level=levels)

        # 对每个系数数组进行阈值处理
        sigma = np.median(np.abs(coeffs[-1])) / 0.0006745
        uthresh = sigma * np.sqrt(2 * np.log(len(part)))
        denoised_coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]

        # 重构数据
        denoised_part = pywt.waverec(denoised_coeffs, wavelet)
        # denoised_data += denoised_part
        if i==0:
            denoised_data += denoised_part
        else:
            denoised_data += denoised_part* 1j

    return toIQ(denoised_data)

if __name__ == '__main__':
    import scipy.io as scio
    from kernal_entroy import diffusion_spectral_entropy
    import random
    import matplotlib.pyplot as plt
    import time

    def awgn(x, snr):
        '''
        加入高斯白噪声 Additive White Gaussian Noise
        :param x: 原始信号
        :param snr: 信噪比
        :return: 加入噪声后的信号
        '''

        np.random.seed(None)  # 设置随机种子
        snr = 10 ** (snr / 10.0)
        xpower = np.sum(np.square(x)) / x.shape[1]
        npower = xpower / snr
        noise = np.zeros((x.shape[0], x.shape[1]))
        if noise.shape[0] == 2:
            noise[0, :] = np.random.randn(x.shape[1]) * np.sqrt(npower)
            noise[1, :] = np.random.randn(x.shape[1]) * np.sqrt(npower)
            return x + noise
        elif noise.shape[0] == 1:
            noise[0, :] = np.random.randn(x.shape[1]) * np.sqrt(npower)
            return x + noise

    data_path = '../data/RML2016_10a_gao17SNRdata.mat'
    data=scio.loadmat(data_path)['data']
    label = scio.loadmat(data_path)['label']
    # data =np.load(data_path)
    sgn=data[0]
    sgnn=awgn(sgn,5)
    sgnn=Mult_Wave_denoise(sgn)
    fig, axs = plt.subplots( 2)
    axs[0].plot(sgn[0], label='I')
    axs[0].plot(sgn[1], label='Q')
    axs[0].legend()
    axs[0].set_xlabel('sgn')
    axs[1].plot(sgnn[0], label='I')
    axs[1].plot(sgnn[1], label='Q')
    axs[1].legend()
    axs[1].set_xlabel('sgnn')
    plt.show()