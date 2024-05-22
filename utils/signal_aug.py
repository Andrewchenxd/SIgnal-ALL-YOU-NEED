import random
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import cv2
from scipy.stats import norm
from scipy.interpolate import CubicSpline
'''
input is IQ signal where shape is [2,128]
'''


def sig_rotate(data):
    new_data = np.zeros_like(data)
    if np.random.random() >= 0.5:
        new_data[1,:] = data[0,:]
        new_data[0,:] = -data[1,:]
    else:
        new_data[1, :] = -data[0, :]
        new_data[0, :] = data[1, :]
    return new_data

def img_rot(img):
    angles=np.random.randint(10, 80)
    img3=np.copy(img)
    img3 = np.transpose(img3, (1, 2, 0))
    rows, cols = img3.shape[:2]
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angles, 1)
    img3 = cv2.warpAffine(img3, M, (cols, rows))
    img3 = np.transpose(img3, (2, 0, 1))
    return img3
def awgn(x, snr=0,zhenshi=False,Seed=1):
    '''
    加入高斯白噪声 Additive White Gaussian Noise
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''

    np.random.seed(Seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.square(x)) / x.shape[1]
    npower = xpower / snr
    noise = np.random.randn(*x.shape)* np.sqrt(npower)

    return x + noise


def rayleigh_noise(x, snr=0,zhenshi=False,Seed=1):
    '''
    加入瑞利噪声
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''

    np.random.seed(Seed)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.square(x)) / x.shape[1]
    npower = xpower / snr
    noise=np.zeros((x.shape[0],x.shape[1]))
    if noise.shape[0]==2:
        noise[0,:] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
        noise[1, :] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
        max_I = np.max(x[0, :])
        max_Q = np.max(x[1, :])
        if zhenshi:
            newsgn = x + noise
            max_In = np.max(newsgn[0, :])
            max_Qn = np.max(newsgn[1, :])
            newsgn[0, :] = newsgn[0, :] * max_I / max_In
            newsgn[1, :] = newsgn[1, :] * max_Q / max_Qn
            return newsgn
        else:
            return x + noise
    elif noise.shape[0]==1:
        noise[0, :] = np.random.rayleigh(np.sqrt(npower), size=x.shape[1])
        return x + noise

def bernoulli_noise(x, snr=0):
    '''
    加入伯努利噪声
    :param x: 原始信号
    :param snr: 信噪比
    :return: 加入噪声后的信号
    '''

    np.random.seed(None)  # 设置随机种子
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(np.square(x)) / x.shape[1]
    npower = xpower / snr
    noise=np.zeros((x.shape[0],x.shape[1]))
    if noise.shape[0]==2:
        noise[0,:] = np.random.binomial(n=1, p=0.5, size=x.shape[1]) * np.sqrt(npower)
        noise[1, :] = np.random.binomial(n=1, p=0.5, size=x.shape[1]) * np.sqrt(npower)
        return x + noise
    elif noise.shape[0]==1:
        noise[0, :] = np.random.binomial(n=1, p=0.5, size=x.shape[1]) * np.sqrt(npower)
        return x + noise


def addmask(sgn,begin=0.1,end=0.5):
    lamba = np.random.uniform(0.1, 0.3, 1)
    mask_num = int(sgn.shape[1] * lamba)
    mask_idx1 = random.sample(range(sgn.shape[1]), mask_num)
    mask_idx2 = random.sample(range(sgn.shape[1]), mask_num)
    # 将需要掩码的元素置为零
    sgn1=np.copy(sgn)
    if sgn1.shape[0]==1:
        sgn1[:, mask_idx1] = 0
    elif sgn1.shape[0] == 2:
        sgn1[0, mask_idx1] = 0
        sgn1[1, mask_idx2] = 0
    return sgn1

def sgndrop(sgn):
    lamba = np.random.uniform(0.1, 0.3, 1)
    mask_num = int(sgn.shape[1] * lamba)
    mask_idx1 = random.sample(range(sgn.shape[1]), mask_num)
    mask_idx2 = random.sample(range(sgn.shape[1]), mask_num)
    # 将需要掩码的元素置为零
    sgn1=np.copy(sgn)
    if sgn1.shape[0]==1:
        sgn1[:, mask_idx1] = 0
    elif sgn1.shape[0] == 2:
        sgn1[0, mask_idx1] = 0
        sgn1[0, mask_idx2] = 0
    return sgn1

def addmask_2d(img,begin=0.1,end=0.5):
    lamba = random.uniform(begin, end)
    mask = np.random.choice([0, 1], size=(1, img.shape[1],  img.shape[2]), p=[lamba, 1-lamba])
    # 将需要掩码的元素置为零
    img1=np.copy(img)
    img1=img1*mask
    return img1

def sig_reserve(data):
    new_data = np.zeros_like(data)
    if np.random.random() <= 0.5:
        new_data[1, :] = data[ 1, ::-1]
        new_data[0, :] = data[ 0, ::-1]
    elif np.random.random() <= 0.75:
        new_data[1, :] = data[1, ::-1]
        new_data[0, :] = data[0, :]
    else:
        new_data[1, :] = data[1, :]
        new_data[0, :] = data[0, ::-1]
    return new_data


def average_pooling(data, pool_size):
    # 获取数据的长度
    length = data.shape[1]

    # 计算需要多少个池来覆盖所有的数据
    num_pools = length // pool_size

    # 创建一个新的数组来保存池化后的数据
    pooled_data = np.zeros((data.shape[0], num_pools))

    # 对每个池进行平均池化
    for i in range(num_pools):
        start = i * pool_size
        end = start + pool_size
        pooled_data[:, i] = np.mean(data[:, start:end], axis=1)

    return pooled_data

def sig_time_warping(data, method='warp', mu=0.0, sigma=0.5, sample_size=2,Num=1):
    '''
    适合对时频图用，时频域变化不大
    :param data: (2,128)
    :return: (2,128)
    '''
    np.random.seed(None)
    new_data = np.copy(data)
    threshold_range = (20, 110)
    for _ in range(Num):  # 只循环一次，所以不需要循环
        threshold = np.random.randint(*threshold_range)
        begin = np.random.randint(128 - threshold - 1)
        end = np.random.randint(begin, 128)

        while end - begin <= threshold:
            end = np.random.randint(begin, 128)

        x = np.arange(begin, end, 1)
        x_new = np.arange(begin, end-(sample_size-1)/sample_size, 1/sample_size)

        f1 = interp1d(x, new_data[0, begin:end], kind='cubic')
        f2 = interp1d(x, new_data[1, begin:end], kind='cubic')

        y1_new = f1(x_new)
        y2_new = f2(x_new)

        downsampled_data = np.zeros([2, y1_new.shape[0]])
        downsampled_data[0] = y1_new
        downsampled_data[1] = y2_new

        if method == 'avg':
            downsampled_data = average_pooling(downsampled_data, sample_size)
        elif method == 'warp':
            L = downsampled_data.shape[1]
            L1 = end - begin - 1
            p = np.random.normal(mu, sigma, L)
            sample = np.random.choice(p, L1, replace=False)
            indices = [np.where(p == element)[0][0] for element in sample]
            sorted_sample = np.sort(indices)
            downsampled_data = downsampled_data[:, sorted_sample]

        new_data[:, begin:end-1] = downsampled_data

    return new_data


def sig_time_warp(x,sigma=0.15,knot=1):
    '''
    适合对信号用，信号变化不大，时频域变化较大
    :param x:
    :param sigma:
    :param knot:
    :return:
    '''
    np.random.seed(None)
    x=np.expand_dims(x,0)
    # 时序偏移
    x = np.swapaxes(x, 1, 2)
    orig_steps = np.arange(x.shape[1])

    random_warps = np.random.normal(loc=1.0, scale=sigma, size=(x.shape[0], knot + 2, x.shape[2]))
    warp_steps = (np.ones((x.shape[2], 1)) * (np.linspace(0, x.shape[1] - 1., num=knot + 2))).T

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        for dim in range(x.shape[2]):
            time_warp = CubicSpline(warp_steps[:, dim], warp_steps[:, dim] * random_warps[i, :, dim])(orig_steps)
            scale = (x.shape[1] - 1) / time_warp[-1]
            ret[i, :, dim] = np.interp(orig_steps, np.clip(scale * time_warp, 0, x.shape[1] - 1), pat[:, dim]).T
    ret = np.swapaxes(ret, 1, 2)
    ret=np.squeeze(ret,0)
    return ret

def sig_time_warping_new(data):
    '''
    :param data: (2,128)
    :return: (2,128)
    '''
    sample_size=8
    new_data=np.copy(data)
    sample_method='jubu'
    if sample_method=='jubu':
        threshold = 110
        begin = np.random.randint(128 - threshold - 1)
        end = np.random.randint(begin, 128)
        # 如果end-begin小于等于阈值，就重新生成end，直到满足条件
        while end - begin <= threshold:
            end = np.random.randint(begin, 128)
        x = np.arange(begin, end, 1)

        # 创建一个新的x轴的坐标数组，从begin到end-1，步长为0.5，用于上采样
        x_new = np.arange(begin, end-(sample_size-1)/sample_size, 1/sample_size)

        # 对new_data的第一行和第二行分别进行插值，使用线性插值方法
        # 忽略越界的值
        f1 = interp1d(x, new_data[0, begin:end], kind='cubic')
        f2 = interp1d(x, new_data[1, begin:end], kind='cubic')

        # 对新的x轴坐标数组进行插值，得到上采样后的结果
        y1_new = f1(x_new)
        y2_new = f2(x_new)
        # downsampled_data=np.zeros([2,y1_new.shape[0]])
        # downsampled_data[0]=y1_new
        # downsampled_data[1] = y2_new
        # downsampled_data = average_pooling(downsampled_data, sample_size)
        new_data[0, begin:end-1] = y1_new[2::sample_size]
        new_data[1, begin:end - 1] = y2_new[2::sample_size]


    return new_data

def sig_convolution(data):
    data = np.expand_dims(data, 0)
    re_data = []
    kernel = 3
    for i in data:
        i_0 = np.zeros_like(i[0])+i[0]
        i_1 = np.zeros_like(i[1])+i[1]
        weight = np.array([0.3,0.4,0.3])
        threshold = 30
        begin = np.random.randint(128 - threshold - 1)
        end = np.random.randint(begin, 128)
        # 如果end-begin小于等于阈值，就重新生成end，直到满足条件
        while end - begin <= threshold:
            end = np.random.randint(begin, 128)
        for j in range(begin, end):
            i_0[j] = (i[0,j-kernel//2:j+kernel//2+1]*weight).sum()
            i_1[j] = (i[1,j-kernel//2:j+kernel//2+1]*weight).sum()

        # plt.plot(i_0)
        # plt.plot(i[0])
        # plt.show()

        i = np.stack([i_0, i_1])
        re_data.append(i)
        re_data = np.array(re_data)
        re_data = np.squeeze(re_data)

    return re_data


def sig_pooling(data):
    data = np.expand_dims(data, 0)
    re_data = []
    kernel = 2
    for i in data:
        i_0 = np.zeros_like(i[0])+i[0]
        i_1 = np.zeros_like(i[1])+i[1]
        threshold = 15
        begin = np.random.randint(128-threshold-1)
        end = np.random.randint(begin, 128)

        # 如果end-begin小于等于阈值，就重新生成end，直到满足条件
        while end - begin <= threshold:
            end = np.random.randint(begin, 128)
        for j in range(begin,end):
            i_0[j] = (i[0,j-kernel//2:j+kernel//2+1]).mean()
            i_1[j] = (i[1,j-kernel//2:j+kernel//2+1]).mean()


        i = np.stack([i_0, i_1])

        re_data.append(i)
        re_data = np.array(re_data)
        re_data = np.squeeze(re_data)


    return re_data




if __name__ == '__main__':
    import scipy.io as scio
    import matplotlib.pyplot as plt
    from signeltoimage import filter
    import time
    fig, axs = plt.subplots(1, 3)
    data_path = '../data/RML2016_10a_gao17SNRdata.mat'
    data = scio.loadmat(data_path)['data']
    label = scio.loadmat(data_path)['label']
    signal = data[648]#648 1223 1850 2443 3023 3609 4198 4790 5407 6023
    signal_aug=np.copy(signal)
    np.random.seed(1)
    random.seed(1)
    # functions = [sig_rotate, addmask, sig_reserve,
    #              sig_time_warping,sig_pooling]
    # # functions=[sig_time_warping]
    # num_select=np.random.randint(0,4)
    # selected_functions = random.sample(functions, 1)
    # for _ in range(1):
    #     for function in selected_functions:
    #         signal_aug = function(signal_aug)
    #
    # s=time.time()
    # for _ in range(100):
    #     _=time_warp(signal_aug)
    # e=time.time()
    # print((e-s)/100)
    signal_aug1 = sig_time_warping(signal_aug,Num=1)
    random_nums1 = np.random.uniform(low=0.9, high=0.99, size=1)
    random_nums2 = np.random.uniform(low=0.5, high=1.5, size=1)
    signal_aug2 = filter(signal_aug , filiter='high', filiter_threshold=0.95,
                                       filiter_size=0.5,return_IQ=True)
    axs[0].plot(signal[0,:], label='I')
    axs[0].plot(signal[1,:], label='Q')
    axs[0].set_title('sgn')
    axs[1].plot(signal_aug1[0, :], label='I')
    axs[1].plot(signal_aug1[1, :], label='Q')
    axs[1].set_title('time warp')
    axs[2].plot(signal_aug2[0, :], label='I')
    axs[2].plot(signal_aug2[1, :], label='Q')
    axs[2].set_title('high frequency Scaling')
    plt.show()


    # img=pwvd(signal)
    # imgm=addmask_2d(img)
    # plt.pcolormesh(imgm[0])
    # plt.show()