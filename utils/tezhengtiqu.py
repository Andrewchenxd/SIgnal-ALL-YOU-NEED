import time
import scipy.io as scio
import scipy.signal
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.special import entr
from scipy.fftpack import fft,fftshift
'''
input (2,128)
'''
def tezheng(sgn):
    # the energy trajectory of a transient signal:
    I = sgn[0,:]
    Q = sgn[1, :]
    len=I.shape[0]
    y=math.sqrt(2)*(I**2+Q**2)/2
    #normalized energy trajectory x(i)
    x=(y-min(y))/(max(y)-min(y)+ 1e-6)
    meanx = np.mean(x)
    varx = np.var(x)
    S = np.mean(((x - meanx) / varx) ** 3)
    K = np.mean(((x - meanx) / varx) ** 4)
    A=sum(x)
    p=np.zeros([len])
    p[0:len-2]=x[1:len-1]
    P=max(p-x)
    #PC
    xxx = np.linspace(0, len/(2*100000), len)
    C = np.polyfit(xxx, np.squeeze(x), 7)
    C0, C1, C2, C3, C4, C5, C6, C7 = C[0], C[1], C[2], C[3], C[4], C[5], C[6], C[7]
    #Entropy
    Entropy=entr(x).sum(axis=0)
    #Maximum value of the power density of the normalizedcentered instantaneous amplitude:
    gama = max(abs(fft(x)))**2/len
    #High-order cumulants
    C42=-2*np.mean((I**2+Q**2))**2
    tez = np.array([S, K,P,A, meanx, varx, C0, C1, C2, C3, C4, C5, C6, C7, C42,gama,Entropy])
    # tez =np.expand_dims(tez, axis=0)
    return tez



# data=scio.loadmat(r'C:\planedemo\电磁论文\data\RML2016.10a_train_data.mat')['data']
# sgn=data[0,:,:]
# strat=time.time()
# a=tezheng(sgn)
# end=time.time()
# print(end-strat)
