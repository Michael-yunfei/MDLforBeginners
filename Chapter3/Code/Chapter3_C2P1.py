# Chapter 3 Lecture 1
# 课堂讲义
# @ Michael

# Import essential packages
# 导入必要扩展

import numpy as np
from scipy import signal

# vector convolution 向量卷积
a = np.array([1, 2, 3])
b = np.array([5, 6, 7, 8, 9])
np.convolve(a, b, 'valid')  # array([34, 40, 46])

# matrix convolution example 1
kernela = np.array([[1, -1], [1, 2]])
datamatrix = np.array([[1, 2, 1], [2, 3, 6], [0, 5, 7]])
print(signal.convolve2d(kernela, datamatrix, 'valid'))
# 注意该方程中的卷积方式，与我们定义的有所不同


def matrixConv(kenl, dtm):
    # Assume size of kenel is less than datamatrix
    # 假设参数核矩阵小于数据矩阵
    m = np.shape(kenl)[0]
    n = np.shape(kenl)[1]
    k = np.shape(dtm)[0]
    h = np.shape(dtm)[1]
    resltmtx = np.zeros([k-m+1, h-n+1])
    for i in range(resltmtx.shape[0]):
        for j in range(resltmtx.shape[1]):
            temp = 0
            for u in range(m):
                for p in range(n):
                    temp += kenl[u, p] * dtm[u+i, p+j]
                    resltmtx[i, j] = temp
    return(resltmtx)


# test the function
matrixConv(kernela, datamatrix)


# matrix convolution example 2
grade_arts = np.ones([6, 6])
grade_science = np.ones([6, 6])
grade_dist1 = [90, 60, 80, 60, 100, 70]
grade_dist2 = [60, 90, 60, 80, 70, 100]
for i in range(len(grade_dist1)):
    grade_arts[:, i] = grade_arts[:, i] * grade_dist1[i]
for j in range(len(grade_dist2)):
    grade_science[:, j] = grade_science[:, j] * grade_dist2[j]

print(grade_arts)
print(grade_science)

kernel_para = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
conv_arts = signal.convolve2d(grade_arts, kernel_para, 'valid')
# array([[-30.,   0.,  60.,  30.],
#        [-30.,   0.,  60.,  30.],
#        [-30.,   0.,  60.,  30.],
#        [-30.,   0.,  60.,  30.]])
signal.convolve2d(conv_arts, kernel_para, 'valid')
# array([[270.,  90.],
#        [270.,  90.]])

# use self-wrote function
conv_arts2 = matrixConv(kernel_para, grade_arts)
# array([[ 30.,   0., -60., -30.],
#        [ 30.,   0., -60., -30.],
#        [ 30.,   0., -60., -30.],
#        [ 30.,   0., -60., -30.]])
matrixConv(kernel_para, conv_arts2)
# array([[270.,  90.],
#        [270.,  90.]])


conv_science = signal.convolve2d(grade_science, kernel_para, 'valid')
conv_science.shape
# array([[  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.]])
signal.convolve2d(conv_science, kernel_para, 'valid')
# array([[ 90., 270.],
#        [ 90., 270.]])

# use self-wrote function
conv_science2 = matrixConv(kernel_para, grade_science)
# array([[  0.,  30., -30., -60.],
#        [  0.,  30., -30., -60.],
#        [  0.,  30., -30., -60.],
#        [  0.,  30., -30., -60.]])
matrixConv(kernel_para, conv_science2)
# array([[ 90., 270.],
#        [ 90., 270.]])
