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
print(signal.convolve2d(datamatrix, kernela, 'valid'))

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


conv_science = signal.convolve2d(grade_science, kernel_para, 'valid')
conv_science.shape
# array([[  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.],
#        [  0., -30.,  30.,  60.]])
signal.convolve2d(conv_science, kernel_para, 'valid')
# array([[ 90., 270.],
#        [ 90., 270.]])
