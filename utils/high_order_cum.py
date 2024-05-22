import taichi as ti
import time
from numba import jit, prange
import numpy as np

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




def calculate_third_order_cumulant(x_input,K):
    # 初始化Taichi
    ti.init()

    # 定义数组的大小
    L = x_input.shape[1]

    # 定义Taichi字段
    x = ti.field(dtype=ti.f32, shape=(2, L))
    C3 = ti.field(dtype=ti.f32, shape=(L, L, L))
    C3_segments = ti.field(dtype=ti.f32, shape=(K, L//K, L//K))

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
                                C3[m, n, k] += x[0, j] * x[0, (j + m) ] * x[0, (j + n) ] * x[0, (j + k) ]
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
    result = np.zeros((K, L//K, L//K), dtype=np.float32)
    for i in range(K):
        for j in range(L//K):
            for k in range(L//K):
                result[i, j, k] = C3_segments[i, j, k]

    # 返回结果
    return result


# 使用随机输入测试函数
x_input = np.random.rand(2, 128).astype(np.float32)
# s=time.time()
# for _ in range(10):
#     result = calculate_third_order_cumulant(x_input,3)
# e=time.time()
# print('gpu time is {}s'.format((e-s)/10))
s=time.time()
for _ in range(10):
    result = segmented_third_order_cumulant(x_input,5)
e=time.time()
print('cpu time is {}s'.format((e-s)/10))