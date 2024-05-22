# 书写人： 高雅晨
# 书写时间： 2022/7/30 22:37
import numpy as np
import math
# from PyEMD import EMD, Visualisation
import scipy.io as scio
from scipy.signal import hilbert
import matplotlib.pyplot as plt
import math
import pandas as pd
from scipy.stats import entropy
from sampen import sampen2

'''
幅值归一化
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
# scio.savemat('./data37/train_data_guiyi.mat', {'data37':new_data})  # -1~1

'''
提取有效信号
'''
# thre=0.5 # 大于等于thre的为有效信号，截取连续
# 提取有效信号
def valid_signal(x,thre=0.5):
    begin=0
    end=0
    for i in range(x.shape[0]): # 从前往后
        if x[i]>=thre:
            begin=i
            break
    for j in range(x.shape[0]-1, -1, -1): # 从后往前
        if x[j]>=thre:
            end=j
            break
    vali_data=x[begin:end+1,:] # 删除幅值小于thre的点
    return vali_data,begin+1,end+1
#画个图看看
# load_mat = scio.loadmat('./data37/train_data_guiyi.mat')
# data37=load_mat['data37']
# a=data37[100,:,:]
# b,begin,end=valid_signal(a,thre)
# plt.figure()
# plt.plot(a)
# plt.title('original signal')
# plt.figure()
# plt.plot(b)
# plt.title('valid signal --- begin={} end={} thre={}'.format(begin,end,thre))
# plt.show()

'''
统计一上一下的
'''
def up_down_count(x):
    x = x.reshape([x.shape[0]])
    counts=0
    for i in range(x.shape[0]):
        if i==x.shape[0]-1:
            break
        if x[i]*x[i+1] < 0:
            counts+=1
    return counts
# load_mat = scio.loadmat('./data37/train_data_guiyi.mat')
# data37=load_mat['data37']
# a=data37[100,:,:]
# b,begin,end=valid_signal(a,thre)
# count=up_down_count(b)
# print(count)
'''
偏度和峰度
'''
def calc_skew_kurt(x):
    s = pd.Series(x.reshape([x.shape[0]]))
    skew=s.skew()  # 偏度
    kurt=s.kurt()  # 峰度
    return skew,kurt
# load_mat = scio.loadmat('./data37/train_data_guiyi.mat')
# data37=load_mat['data37']
# a=data37[100,:,:]
# skew,kurt=calc_skew_kurt(a)
# print(skew,kurt)

'''
波形的熵  信息熵、样本熵
'''
# 信息熵
def calc_entropy(x):
    x=x.reshape([x.shape[0]])
    entro=entropy(x,base=2)  # 默认e为底，可指定2为底
    return entro
# load_mat = scio.loadmat('./data37/train_data_guiyi.mat')
# data37=load_mat['data37']
# a=data37[100,:,:]
# entro=calc_entropy(a)
# print(entro)  # -inf ???

# 样本熵
def calc_samen(x):
    x = x.reshape([x.shape[0]])
    sampen_of_series = sampen2(x) # list3  # Epoch length for max epoch（最大长度）SampEn（样本熵的值）Standard Deviation（标准偏差）
    return sampen_of_series[-1][1]
# load_mat = scio.loadmat('./data37/train_data_guiyi.mat')
# data37=load_mat['data37']
# a=data37[100,:,:]
# entro=calc_samen(a)
# print(entro)
# data37=scio.loadmat('./train/RML2016.10a_train_data.mat')['data37']
# te=[]
# for i in range(40000):
#     sgn=data37[i,:]
#     sgn=normalized(sgn)
#     b,begin,end=valid_signal(sgn,thre)
#     count = up_down_count(b)
#     te.append(count)
#
# te=np.array(te)

