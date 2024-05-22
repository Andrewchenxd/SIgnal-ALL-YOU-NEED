# 书写人： 高雅晨
# 书写时间： 2022/7/30 17:02
import numpy as np
import math
# from PyEMD import EMD, Visualisation
import scipy.io as scio
from scipy.signal import hilbert
import matplotlib.pyplot as plt

emd_num=7

'''
step1:数据预处理——幅值归一化
'''
def normalized(x):
    y = x - np.mean(x)  # 消除直流分量
    y = y / np.max(np.abs(y))  # 幅值归一化
    return (y)
# load_mat = scio.loadmat('./data37/RML2016.10a_train_data.mat')
# ori_data=load_mat['data37']
# new_data=np.zeros([ori_data.shape[0],ori_data.shape[1],ori_data.shape[2]])
# for i in range(ori_data.shape[0]):
#     new_data[i,:,:]=normalized(ori_data[i,:,:])
#
# scio.savemat('./data37/train_data_guiyi.mat', {'data37':new_data})
'''
step2:按EMD算法对预处理后的信号进行HHT变换，得到信号HHT时频谱
'''
def emd_visual(x,order):
    x=x.reshape([x.shape[0]])
    t= np.arange(0,x.shape[0], 1)
    emd = EMD()
    if order=='zhiding':
        imfs= emd.emd(x, max_imf=emd_num-1)  # 指定最多分量
        # 绘制 IMF
        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, include_residue=False)
        # 绘制并显示所有提供的IMF的瞬时频率
        vis.plot_instant_freq(imfs=imfs,t=t)
        vis.show()
    if order=='weizhiding':  #8个imf
        emd.emd(x)
        imfs, res = emd.get_imfs_and_residue() #未指定最多分量
        # 绘制 IMF
        vis = Visualisation()
        vis.plot_imfs(imfs=imfs, residue=res, include_residue=True)
        # 绘制并显示所有提供的IMF的瞬时频率
        vis.plot_instant_freq(imfs=imfs,t=t)
        vis.show()

def get_emd(x,method=1):

    x = x.reshape([x.shape[1]])
    emd = EMD()
    if method==0:
        emd.emd(x)
        imfs, res = emd.get_imfs_and_residue()  # 未指定最多分量
    if method==1:
        imfs = emd.emd(x, max_imf=emd_num - 1)  # 指定最多分量的emd
    return imfs


# 希尔波特变换及画时频谱
# 将信号EMD分解所得各IMF分量imf_i(t)进行Hilbert变换得到R_i(f,t)，然后R_i(f,t)对i求和得到HHT时频谱TF(f,t)
def hhtlw(IMFs, t, f_range, t_range, ft_size, draw):
    fmin, fmax = f_range[0], f_range[1]  # 时频图所展示的频率范围
    tmin, tmax = t_range[0], t_range[1]  # 时间范围
    fdim, tdim = ft_size[0], ft_size[1]  # 时频图的尺寸（分辨率）
    dt = (tmax - tmin) / (tdim - 1)
    df = (fmax - fmin) / (fdim - 1)
    vis = Visualisation()
    # 希尔伯特变化
    c_matrix = np.zeros((fdim, tdim))
    for imf in IMFs:
        imf = np.array([imf])
        # 求瞬时频率
        freqs = abs(vis._calc_inst_freq(imf, t, order=False, alpha=None))
        # 求瞬时幅值
        amp = abs(hilbert(imf))
        # 去掉为1的维度
        freqs = np.squeeze(freqs)
        amp = np.squeeze(amp)
        # 转换成矩阵
        temp_matrix = np.zeros((fdim, tdim))
        n_matrix = np.zeros((fdim, tdim))
        for i, j, k in zip(t, freqs, amp):
            if i >= tmin and i <= tmax and j >= fmin and j <= fmax:
                temp_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += k
                n_matrix[round((j - fmin) / df)][round((i - tmin) / dt)] += 1
        n_matrix = n_matrix.reshape(-1)
        idx = np.where(n_matrix == 0)[0]
        n_matrix[idx] = 1
        n_matrix = n_matrix.reshape(fdim, tdim)
        temp_matrix = temp_matrix / n_matrix
        c_matrix += temp_matrix

    t = np.linspace(tmin, tmax, tdim)
    f = np.linspace(fmin, fmax, fdim)
    # 可视化
    # if draw == 1:
    #     fig, axes = plt.subplots()
    #     plt.rcParams['font.sans-serif'] = 'Times New Roman'
    #     plt.contourf(t, f, c_matrix, cmap="jet")
    #     plt.xlabel('Time/s', fontsize=16)
    #     plt.ylabel('Frequency/Hz', fontsize=16)
    #     plt.title('Hilbert spectrum', fontsize=20)
    #     x_labels = axes.get_xticklabels()
    #     [label.set_fontname('Times New Roman') for label in x_labels]
    #     y_labels = axes.get_yticklabels()
    #     [label.set_fontname('Times New Roman') for label in y_labels]
    #     plt.show()
    return t, f, c_matrix


# data37 = scio.loadmat('./train/RML2016.10a_train_data.mat')['data37']
#
# a=data37[5,:,:]
# imfs=get_emd(a,1)
# tt,ff,c_matrix=hhtlw(imfs,t=np.arange(0, 1.0, 1.0 / a.shape[0]),f_range=[0,2000],t_range=np.arange(0,a.shape[0], 1),ft_size=[128,128],draw=1)
