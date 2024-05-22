import torch
from scipy.signal import spectrogram
import numpy as np
from tftb.processing import *
import cv2
import pywt
# from scipy.signal import kaiser, hamming
from scipy.signal import resample
from pyts.image import GramianAngularField
from numba import jit, prange
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numba as nb
'''
实现功能：输入2维雷达信号sgn，大小如（1,2999），进行短时傅里叶变换
'''
import pyfftw

def resampe(sgn,samplenum=2):
    sgn = resample(sgn, sgn.shape[1] // samplenum, axis=1)
    return sgn



def stp(data,resize_is=False,nperseg=42,lap=40,norm='maxmin',resize_num=128):
    t = np.linspace(0, 1, 10001)
    fs = 1 / (t[1] - t[0])
    sgn=data
    if sgn.shape[0]==2:
        sgn = data[0, :] + data[1, :] * 1j

    f, tt, p = spectrogram(x=sgn,fs=fs, nperseg=nperseg,
                               noverlap=lap,
                               return_onesided=False,nfft=128)

    if norm=='linalg':
        normImg = p/np.linalg.norm(p)
    elif norm=='maxmin':
        normImg = p
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min())
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(p)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'denoise':
        normImg = p / np.linalg.norm(p)
        # normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() )
        Max=np.max(normImg)
        Min=np.min(normImg)
        yuzhi=(9/10)*np.percentile(normImg,25)+(1/10)*(Max+Min)
        normImg =np.where(normImg>yuzhi,normImg ,0)
    if resize_is==True:
        normImg = cv2.resize(normImg, (resize_num,resize_num))
        normImg = np.expand_dims(normImg, axis=0)
    else:
        normImg = np.expand_dims(normImg, axis=0)

    return normImg

def converet_RGB(img,cmap='rainbow'):
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=img.min(), vmax=img.max())
    rgba_data = cmap(norm(img))
    # rgba_data = cmap(img)
    rgba_data = np.delete(rgba_data, 3, axis=2)
    rgb_data = np.transpose(rgba_data, (2, 0, 1))
    return rgb_data

def pwvd(data,norm='no',resize_is=False,resize_num=128,RGB_is=False,cmap='rainbow'):
    t = np.linspace(0, 1, data.shape[1])
    sgn = data[0, :] + data[1, :] * 1j
    spec = PseudoWignerVilleDistribution(sgn, timestamps=t)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    img,_,_=spec.run()

    if resize_is==True:
        img=cv2.resize(img, (resize_num,resize_num))
    if norm=='maxmin':
        # img = np.abs(np.log(1 + np.abs(img)))
        ampLog=img
        img = (ampLog - ampLog.min()) / (ampLog.max() - ampLog.min())

    if norm == 'meanvar':
        img = (img - np.mean(img)) / (np.std(img) + 1.e-9)
    if norm == 'maxmin2':
        Min=-0.000999912436681158
        Max=0.0013195788363637715
        img = (img - Min) / (Max - Min +1e-6)
    if norm == 'maxmin2018':
        Min=-69.02861173465756
        Max=423.1396374466868
        img = (img - Min) / (Max - Min +1e-6)
    if norm == 'meanstd':
        mean=7.318065201053581e-05
        std=0.00025146489151817126
        img=(img-mean)/std
    if RGB_is==True:
        img=converet_RGB(img,cmap=cmap)
    if RGB_is == False:
        img=np.expand_dims(img, axis=0)

    return img

def gasf(data,norm='no',resize_is=False,resize_num=128,RGB_is=False,sample_range=None,cmap='rainbow'):
    gaf = GramianAngularField(method='summation',sample_range=sample_range, overlapping=True)
    # sgn0=np.expand_dims(data,0)
    # sgn1 = np.expand_dims(data[1], 0)
    img0 = gaf.fit_transform(data)
    # img1 = gaf.fit_transform(sgn1)
    img = np.mean(img0,0)
    # img=np.squeeze(img,0)



    if resize_is==True:
        img=cv2.resize(img, (resize_num,resize_num))
    if norm=='maxmin':
        # img = np.abs(np.log(1 + np.abs(img)))
        for i in range(img.shape[0]):
            img[i] = (img[i] - img[i].min()) / (img[i].max() - img[i].min())

    if norm == 'meanstd':
        for i in range(img.shape[0]):
            img[i] = (img[i] - np.mean(img[i])) / (np.std(img[i]) + 1.e-9)

    if RGB_is==True:
        img=converet_RGB(img, cmap=cmap)

    if RGB_is == False:
        img = np.expand_dims(img, axis=0)

    return img

# data=scio.loadmat(r'C:\planedemo\电磁论文\data\RML2016.10a_train_data.mat')['data']
# sgn=data[0,:,:]
# a=psvd(sgn)

def cwt(x, fs, totalscal, wavelet='morl'):
    if wavelet not in pywt.wavelist():
        print('小波函数名错误')
    else:
        wfc = pywt.central_frequency(wavelet=wavelet)
        a = 2 * wfc * totalscal/(np.arange(totalscal,0,-1))
        period = 1.0 / fs
        [cwtmar, fre] = pywt.cwt(x, a, wavelet, period)
        amp = abs(cwtmar)
        return amp, fre


def wave(data,resampe_is=False,samplenum=2,mask_is=False,resize_is=False,fs=128,scales=128,norm='maxmin',resize_num=128):

    sgn=data
    if resampe_is==True:
        sgn=resampe(sgn,samplenum=samplenum)
    img, fre = cwt(sgn, fs, scales, 'morl')
    img=np.mean(img,axis=1)
    # spec = PseudoWignerVilleDistribution(sgn)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)
    if norm=='maxmin':
        normImg = (img - img.min()) / (img.max() - img.min() +0.000001)
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'lognew':
        normImg = (np.log(np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)

    if resize_is==True:
        normImg = cv2.resize(normImg, (resize_num,resize_num))
    normImg = np.expand_dims(normImg, axis=0)
    return normImg


def wave1(data,resize_is=False,fs=1024,scales=512,norm='log'):
    sgn=data
    if sgn.shape[0]==2:
        sgn = data[0, :] + data[1, :] * 1j
        sgn=np.expand_dims(sgn,0)
    img, fre = cwt(sgn[0], fs, scales, 'morl')
    # spec = PseudoWignerVilleDistribution(sgn)
    # spec = smoothed_pseudo_wigner_ville(sgn, timestamps=t)a
    if norm=='maxmin':
        normImg = (img - img.min()) / (img.max() - img.min() +0.000001)
    elif norm=='log':
        normImg = np.abs(np.log(1.000001 + np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    elif norm == 'lognew':
        normImg = (np.log(np.abs(img)))
        normImg = (normImg - normImg.min()) / (normImg.max() - normImg.min() + 0.000001)
    if resize_is==True:
        normImg = cv2.resize(normImg, (128,128))
    normImg = np.expand_dims(normImg, axis=0)
    return normImg

# def spwvd(data,norm='no',resize_is=False):
#     twindow = hamming(13)
#     fwindow = hamming(33)
#     sgn = data[0, :] + data[1, :] * 1j
#     img = smoothed_pseudo_wigner_ville(sgn, twindow=twindow, fwindow=fwindow,
#                                        freq_bins=128)
#     if resize_is==True:
#         img=cv2.resize(img, (128,128))
#     if norm=='maxmin':
#         # img = np.abs(np.log(1 + np.abs(img)))
#         ampLog=img
#         img = (ampLog - ampLog.min()) / (ampLog.max() - ampLog.min())
#
#     if norm == 'meanvar':
#         img = (img - np.mean(img)) / (np.var(img) + 1.e-9) ** .5
#
#     img = np.expand_dims(img, axis=0)
#     return img


@jit(nopython=True)
def forth_order_cumulant(x):
    L = len(x)
    C4 = np.zeros((L, L, L, L))
    rolls = [np.roll(x, -i) for i in range(L)]  # 预先计算滚动数组
    for m in range(L):
        for n in range(L):
            for k in range(L):
                for j in range(L):
                    C4[m, n, k, j] = np.mean(x * rolls[m] * rolls[n] * rolls[k]* rolls[j])
    return C4

@jit(nopython=True)
def third_order_cumulant(x):
    L = len(x)
    C3 = np.zeros((L, L, L))
    rolls = [np.roll(x, -i) for i in range(L)]  # 预先计算滚动数组
    for m in range(L):
        for n in range(L):
            for k in range(L):
                C3[m, n, k] = np.mean(x * rolls[m] * rolls[n] * rolls[k])
    return C3

@jit(nopython=True)
def segmented_third_order_cumulant(x, K,IQsum=False):
    dim=x.shape[0]
    L = x.shape[1]
    if IQsum:
        x[0, :]=x[0, :]+x[1, :]
    segment_length = L // K
    C3_segments = np.zeros((K, segment_length, segment_length))  # 使用元组作为形状参数
    if IQsum:
        for i in prange(K):
            start = i * segment_length
            end = start + segment_length
            segment = x[0,start:end]
            if len(segment) >= 3:
                C3 = third_order_cumulant(segment)
                for m in range(segment_length):
                    for n in range(segment_length):
                        C3_segments[i, m, n] = np.mean(C3[:, m, n])/2
    else:
        for d in prange(dim):
            for i in prange(K):
                start = i * segment_length
                end = start + segment_length
                segment = x[d,start:end]
                if len(segment) >= 3:
                    C3 = third_order_cumulant(segment)
                    for m in range(segment_length):
                        for n in range(segment_length):
                            C3_segments[i, m, n] = np.mean(C3[:, m, n])/2  # 使用循环来计算沿第0轴的平均值

    return C3_segments

@jit(nopython=True)
def segmented_forth_order_cumulant(x, K,IQsum=False):
    dim=x.shape[0]
    L = x.shape[1]
    if IQsum:
        x[0, :]=x[0, :]+x[1, :]
    segment_length = L // K
    C4_segments = np.zeros((K, segment_length, segment_length))  # 使用元组作为形状参数
    if IQsum:
        for i in prange(K):
            start = i * segment_length
            end = start + segment_length
            segment = x[0,start:end]
            if len(segment) >= 3:
                C4 = forth_order_cumulant(segment)
                for m in range(segment_length):
                    for n in range(segment_length):
                        C4_segments[i, m, n] = np.mean(C4[:, m, n])/2
    else:
        for d in prange(dim):
            for i in prange(K):
                start = i * segment_length
                end = start + segment_length
                segment = x[d,start:end]
                if len(segment) >= 3:
                    C4 = forth_order_cumulant(segment)
                    for m in range(segment_length):
                        for n in range(segment_length):
                            C4_segments[i, m, n] = np.mean(C4[:, m, n,:])/2  # 使用循环来计算沿第0轴的平均值

    return C4_segments


def angle_phase(sgn):
    complex_signal = np.complex128(sgn[0]+1j*sgn[1])
    amplitude = np.abs(complex_signal)
    phase = np.angle(complex_signal)
    return amplitude, phase

def filter(sgn,filiter='high',filiter_threshold=0.99,filiter_size=0.0,middle_zero=False,freq_smooth=False,return_IQ=False):
    I = sgn[0]
    Q = sgn[1]
    IQ = I + 1j * Q  # 复数形式的IQ数据

    # 对复数数据进行傅里叶变换
    N = len(IQ)
    IQ_fft = np.fft.fft(IQ,n=N)
    IQ_abs = np.abs(IQ_fft)
    sorted_indices = np.argsort(IQ_abs)
    if filiter=='high':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs >= threshold] *= filiter_size  # 将大于阈值的点设为0
    elif filiter == 'low':
        threshold_index = int(filiter_threshold * N)  # 计算排名前20%的索引
        threshold = IQ_abs[sorted_indices[threshold_index]]  # 找到阈值
        IQ_fft[IQ_abs <= threshold] *= filiter_size  # 将大于阈值的点设为0
        if middle_zero:
            IQ_fft[20:110]=0.001
        if freq_smooth:
            # 定义平滑窗口的大小
            window_size = 3

            # 创建一个新的数组，用于存储平滑后的数据
            smoothed_arr = np.zeros_like(IQ_fft)

            # 对数组进行平滑处理
            for i in range(window_size, len(IQ_fft) - window_size):
                smoothed_arr[i] = np.mean(IQ_fft[i - window_size:i + window_size])
    sgn_IQ = np.copy(sgn)
    sgn = np.fft.ifft(IQ_fft)
    if return_IQ:
        sgn_IQ[0]=np.real(sgn)
        sgn_IQ[1] = np.imag(sgn)
        return sgn_IQ
    else:
        return sgn

def sgn_norm(sgn,normtype='maxmin'):
    if normtype=='maxmin':
        sgn = (sgn - sgn.min()) / (sgn.max() - sgn.min())
    elif normtype == 'maxmin-1':
        sgn = (2*sgn - sgn.min()- sgn.max()) / (sgn.max() - sgn.min())
    else:
        sgn=sgn
    return sgn


def moving_avg_filter(signal, window_size):
    """
    Applies a moving average filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)

    # Apply the moving average filter to each channel
    for channel in range(2):
        # Pad the signal with zeros to handle boundary conditions
        padded_signal = np.pad(signal[channel], (window_size // 2, window_size - 1 - window_size // 2), mode='edge')

        for i in range(signal.shape[1]):
            # Calculate the moving average using the window
            filtered_signal[channel, i] = np.sum(padded_signal[i:i + window_size]) / window_size

    return filtered_signal


def gaussian_filter(signal, sigma=1, kernel_radius=7):
    """
    Applies a Gaussian filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        sigma (float): Standard deviation of the Gaussian kernel.
        kernel_radius (int): Radius of the Gaussian kernel.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)

    # Create the Gaussian kernel
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    # Normalize the Gaussian kernel
    # gaussian_kernel /= np.sum(gaussian_kernel)

    # Apply the Gaussian filter to each channel
    for channel in range(2):
        # Pad the signal with zeros to handle boundary conditions
        padded_signal = np.pad(signal[channel], kernel_radius, mode='edge')

        for i in range(signal.shape[1]):
            # Convolve the signal with the Gaussian kernel
            filtered_signal[channel, i] = np.convolve(padded_signal, gaussian_kernel, mode='valid')[i]

    return filtered_signal



@nb.jit(nopython=True)
def moving_avg_filter_numba(signal, window_size):
    """
    Applies a moving average filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        window_size (int): Size of the moving average window.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)
    L = signal.shape[1]

    # Apply the moving average filter to each channel
    for channel in range(2):
        padded_signal = np.zeros(L + window_size - 1, dtype=signal.dtype)
        padded_signal[:window_size // 2] = signal[channel, 0]  # Pad left with the first value
        padded_signal[-window_size // 2:] = signal[channel, -1]  # Pad right with the last value
        padded_signal[window_size // 2:window_size // 2 + L] = signal[channel]

        for i in range(L):
            # Calculate the moving average using the window
            filtered_signal[channel, i] = np.sum(padded_signal[i:i + window_size]) / window_size

    return filtered_signal

@nb.jit(nopython=True)
def gaussian_filter_numba(signal, sigma=1, kernel_radius=7):
    """
    Applies a Gaussian filter to the input signal.

    Args:
        signal (numpy.ndarray): Input signal with shape (2, L), where 2 represents I/Q channels and L is the signal length.
        sigma (float): Standard deviation of the Gaussian kernel.
        kernel_radius (int): Radius of the Gaussian kernel.

    Returns:
        numpy.ndarray: Filtered signal with the same shape as the input signal.
    """
    # Check if the input signal has the expected shape
    if signal.shape[0] != 2:
        raise ValueError("Input signal must have shape (2, L), where 2 represents I/Q channels.")

    filtered_signal = np.zeros_like(signal)
    L = signal.shape[1]
    kernel_size = 2 * kernel_radius + 1

    # Create the Gaussian kernel
    x = np.arange(-kernel_radius, kernel_radius + 1)
    gaussian_kernel = np.exp(-0.5 * (x / sigma) ** 2) / (np.sqrt(2 * np.pi) * sigma)

    # Normalize the Gaussian kernel
    gaussian_kernel /= np.sum(gaussian_kernel)

    # Apply the Gaussian filter to each channel
    for channel in range(2):
        padded_signal = np.zeros(L + 2 * kernel_radius, dtype=signal.dtype)
        padded_signal[:kernel_radius] = signal[channel, 0]  # Pad left with the first value
        padded_signal[-kernel_radius:] = signal[channel, -1]  # Pad right with the last value
        padded_signal[kernel_radius:kernel_radius + L] = signal[channel]

        for i in range(L):
            # Convolve the signal with the Gaussian kernel
            filtered_signal[channel, i] = 0.0
            for j in range(kernel_size):
                filtered_signal[channel, i] += padded_signal[i + j] * gaussian_kernel[j]

    return filtered_signal

if __name__ == '__main__':
    from scipy.stats import entropy
    import scipy.io as scio
    # from kernal_entroy import diffusion_spectral_entropy
    import random
    import matplotlib.pyplot as plt
    from tezhengtiqu import tezheng
    from sklearn.manifold import TSNE
    from matplotlib.colors import ListedColormap
    import time
    import matplotlib.colors as colors
    from imgaug import augmenters as iaa
    import matplotlib
    from sklearn.decomposition import PCA,IncrementalPCA
    import taichi as ti
    import pickle
    from signal_aug import *
    from matplotlib.colors import LinearSegmentedColormap
    matplotlib.use('TkAgg')
    # matplotlib.use('Agg')
    def awgn(x, snr=0, zhenshi=False):
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
            max_I = np.max(x[0, :])
            max_Q = np.max(x[1, :])
            min_I = np.min(x[0, :])
            min_Q = np.min(x[1, :])
            if zhenshi:
                newsgn = x + noise
                max_In = np.max(newsgn[0, :])
                max_Qn = np.max(newsgn[1, :])
                min_In = np.min(newsgn[0, :])
                min_Qn = np.min(newsgn[1, :])
                newsgn[0, :] = newsgn[0, :] * (max_I-min_I) / (max_In-min_In)
                newsgn[1, :] = newsgn[1, :] * (max_Q-min_Q) / (max_Qn-min_Qn)
                return newsgn
            else:
                return x + noise
        elif noise.shape[0] == 1:
            noise[0, :] = np.random.randn(x.shape[1]) * np.sqrt(npower)
            return x + noise


    def addmask(sgn, lamba):
        mask_num = int(sgn.shape[1] * lamba)
        mask_idx1 = random.sample(range(sgn.shape[1]), mask_num)
        mask_idx2 = random.sample(range(sgn.shape[1]), mask_num)
        # 将需要掩码的元素置为零
        sgn1 = np.copy(sgn)
        if sgn1.shape[0] == 1:
            sgn1[:, mask_idx1] = 0
        elif sgn1.shape[0] == 2:
            sgn1[0, mask_idx1] = 0
            sgn1[1, mask_idx2] = 0
        return sgn1


    def rayleigh_noise(x, snr=0):
        '''
        加入瑞利噪声
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
            noise[0, :] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
            noise[1, :] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
            return x + noise
        elif noise.shape[0] == 1:
            noise[0, :] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
            return x + noise

    def addmask_2d(img, lamba):
        mask = np.random.choice([0, 1], size=(1, img.shape[1], img.shape[2]), p=[lamba, 1 - lamba])
        # 将需要掩码的元素置为零
        img1 = np.copy(img)
        img1 = img1 * mask
        return img1

    def count_shannon2(img):
        # 把矩阵的值映射到0-255的整数范围
        max_b = np.max(img,axis=0)
        min_b = np.min(img,axis=0)
        # # 归一化tensor，使其在[0,1]之间
        img = (img - min_b) / (max_b - min_b + 1e-8)
        img=img.flatten()
        img = (img * 255).astype(np.uint8)

        # 计算图像的直方图，即每个灰度值出现的频率
        hist, bins = np.histogram(img, bins=256, range=(0, 255), density=True)

        # 计算直方图的信息熵，即每个灰度值出现的不确定性
        shannon = entropy(hist)
        return shannon

    def count_var(tsne_img1,label_sgn):
        # 获取所有的类别
        labels = np.unique(label_sgn)

        # 初始化类内方差和类间方差
        within_class_variances = []
        between_class_variances = []

        # 计算每个类别的类内方差和类间方差
        for label in labels:
            # 获取当前类别的数据
            class_data = tsne_img1[label_sgn == label]

            # 计算类内方差
            within_class_variance = np.var(class_data, axis=0)
            within_class_variances.append(within_class_variance)

            # 计算类间方差
            between_class_variance = np.var(tsne_img1 - np.mean(class_data, axis=0), axis=0)
            between_class_variances.append(between_class_variance)

        # 将结果转换为numpy数组
        within_class_variances = np.array(within_class_variances)
        between_class_variances = np.array(between_class_variances)
        return  np.sum(within_class_variances,1), np.sum(between_class_variances,1)


    def calculate_third_order_cumulant(x_input, K):
        # 初始化Taichi
        ti.init()

        # 定义数组的大小
        L = x_input.shape[1]

        # 定义Taichi字段
        x = ti.field(dtype=ti.f32, shape=(2, L))
        C3 = ti.field(dtype=ti.f32, shape=(L, L, L))
        C3_segments = ti.field(dtype=ti.f32, shape=(K, L // K, L // K))

        @ti.kernel
        def segmented_third_order_cumulant():
            segment_length = L // K
            for i in range(K):
                start = i * segment_length
                end = start + segment_length
                if segment_length >= 3:
                    for m in range(L):
                        for n in range(L):
                            for k in range(L):
                                for j in range(start, end):
                                    C3[m, n, k] += x[0, j] * x[0, (j + m)] * x[0, (j + n)] * x[0, (j + k)]
                                    C3[m, n, k] += x[1, j] * x[1, (j + m)] * x[1, (j + n)] * x[1, (j + k)]
                                C3[m, n, k] /= segment_length
                    for m in range(segment_length):
                        for n in range(segment_length):
                            for k in range(segment_length):
                                C3_segments[i, m, n] += C3[k, m, n] / segment_length

        # 将输入数据复制到Taichi字段
        for i in range(2):
            for j in range(L):
                x[i, j] = x_input[i, j]

        # 调用Taichi内核
        # third_order_cumulant()
        segmented_third_order_cumulant()

        # 将结果从Taichi字段复制到numpy数组
        result = np.zeros((K, L // K, L // K), dtype=np.float32)
        for i in range(K):
            for j in range(L // K):
                for k in range(L // K):
                    result[i, j, k] = C3_segments[i, j, k]

        # 返回结果
        return result


    def pca_on_image(img,K=128):
        total_size = np.prod(img.shape)
        # 将图像展平成一维向量
        new_shape = (K, total_size // K)
        img_flattened = img.reshape(new_shape)

        # 创建 PCA 对象，n_components 参数表示降维后的维度
        pca = PCA(n_components=K)

        # 对展平后的图像进行 PCA
        img_pca = pca.fit_transform(img_flattened)

        return img_pca

    def torch_stp(sgn,resize_is=False):
        sgn=torch.unsqueeze(sgn,0)
        I = sgn[:, 0, :]
        Q = sgn[:, 1, :]
        # freq = torch.abs(torch.fft.fft(I + Q * 1j))
        freq = torch.abs(torch.stft((I + Q* 1j) , n_fft=128, hop_length=3, win_length=4, window=None
                                   ,return_complex=False))
        freq = freq.sum(-1)

        if resize_is==False:
            return freq.squeeze(1)
        else:
            freq_resized = torch.nn.functional.interpolate(freq.unsqueeze(1), size=(128, 128), mode='bicubic')

            freq_resized = freq_resized.squeeze(1)
            return freq_resized



    data_path = '../data/RML2016_10a_gao17SNRdata.mat'
    # data_path2 = '../dat/'
    path='../data/ADSB.pkl'
    with open(path, 'rb') as f:
        # 从文件中反序列化出数据字典
        dataset = pickle.load(f)

    # 提取出data和label数组
    data=dataset['data']
    labels=dataset['label']
    sgn=data[10000]
    sgn=resampe(sgn,3)
    # sgn=sgn[:,0:200]
    sgn_aug=sgn
    sgn_n=awgn(sgn,0)
    window_size=9
    sigma = 2
    kernel_radius = 7
    sgn_fliter = moving_avg_filter_numba(sgn_n, window_size=window_size)
    sgn_fliter = gaussian_filter_numba(sgn_fliter, sigma=sigma, kernel_radius=kernel_radius)
    t = np.arange(0, sgn.shape[1], 1)  # 时间序列
    plt.subplot(2, 3, 1)
    plt.plot(t, sgn[0])
    plt.plot(t, sgn[1])
    plt.title('sgn_org')
    plt.subplot(2, 3, 2)
    plt.plot(t, sgn_n[0])
    plt.plot(t, sgn_n[1])
    plt.title('sgn')
    plt.subplot(2, 3, 3)
    plt.plot(t, sgn_fliter[0])
    plt.plot(t, sgn_fliter[1])
    plt.show()
    plt.title('sgn_new')
    s = time.time()
    for _ in range(100):
        # random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
        # random_nums2 = np.random.uniform(low=0.4, high=1.6, size=1)
        # sgn_aug = filter(sgn_aug, filiter='high', filiter_threshold=random_nums1,
        #                       filiter_size=random_nums2, return_IQ=True)
        sgn_fliter = moving_avg_filter_numba(sgn_n, window_size=window_size)
        sgn_fliter = gaussian_filter_numba(sgn_fliter, sigma=sigma, kernel_radius=kernel_radius)
    e=time.time()
    print((e-s)/100)




