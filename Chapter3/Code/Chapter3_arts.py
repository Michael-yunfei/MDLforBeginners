# 第三章 文科班 讲义代码
# @ Michael

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mtmg


# set the working direction
os.getcwd()
os.chdir('/Users/Michael/Documents/MDLforBeginners/Chapter3/Code')

# read the image
im_fish = mtmg.imread('Images/CastiFish.jpg')
im_fish.shape
plt.imshow(im_fish[:, :, 2])
im_fish[2:100, 2:100, 0]

# construct an 64 by 64, RGB(187, 161, 112)
im_cons1 = np.ones([64, 64, 3])
im_cons1[:, :, 0] = 187 * np.ones([64, 64])
im_cons1[:, :, 1] = 161 * np.ones([64, 64])
im_cons1[:, :, 2] = 112 * np.ones([64, 64])
im_cons1 = im_cons1.astype(int)
plt.imshow(im_cons1)


# Conv1 function, edge detection
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


# conv2 function, pooling
def matrixPooling(kenl, dtm):
    """
    A function that is doing pooling on matrix dtm
    Input: kenl - parameter for pooling, like every 2 by 2 or 3 by 3 square
           dtm - matrix with square dimension, like 64 by 64
    output: a smaller matrix with the maxium of value of each sub square matrix
    """

    m = dtm.shape[0]
    n = dtm.shape[1]

    if m != n or m % kenl != 0:
        raise ValueError('Input matrix is not square or m modulo kenl is \
                         not equal to zero')
    output_size = int(m / kenl)
    output_mat = np.zeros([output_size, output_size])
    for i in range(output_size):
        for j in range(output_size):
            output_mat[i, j] = np.max(dtm[i*kenl:(i+1)*kenl,
                                          j*kenl:(j+1)*kenl])

    return output_mat


# convolution with chinese character (64 by 64)
im_three = np.ones([64, 64, 3]) * 255
im_three[5:8, 3:60, :] = 32 * np.ones([3, 57, 3])
im_three[28:31, 12:50, :] = 32 * np.ones([3, 38, 3])
im_three[56:59, 3:60, :] = 32 * np.ones([3, 57, 3])
im_three = im_three.astype(int)
plt.imshow(im_three)

kernel_para_h = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
im_three_con = np.ones([62, 62, 3])
for i in range(3):
    im_three_con[:, :, i] = matrixConv(kernel_para_h, im_three[:, :, i])
im_three_con = im_three_con.astype(int)
plt.imshow(im_three_con)

# pooling the image
im_three_con_pooling = np.ones([31, 31, 3])
for i in range(3):
    im_three_con_pooling[:, :, i] = matrixPooling(2, im_three_con[:, :, i])
im_three_con_pooling = im_three_con_pooling.astype(int)
plt.imshow(im_three_con_pooling)


# read another chinese character
im_chuan = mtmg.imread('Images/chuan.jpg')
im_chuan.shape  # 774 720 3
im_chuan = im_chuan[:720, :, :]
im_chuan.shape
plt.imshow(im_chuan)

# convolution with vertical edge detection
kernel_para_v = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
im_chuan_con = np.ones([718, 718, 3])
for i in range(3):
    im_chuan_con[:, :, i] = matrixConv(kernel_para_v, im_chuan[:, :, i])
im_chuan_con = im_chuan_con.astype(int)
plt.imshow(im_chuan_con)

# pooling im_chuan_con
im_chuan_con_pooling = np.ones([int(718/2), int(718/2), 3])
for i in range(3):
    im_chuan_con_pooling[:, :, i] = matrixPooling(2, im_chuan_con[:, :, i])
im_chuan_con_pooling = im_chuan_con_pooling.astype(int)
plt.imshow(im_chuan_con_pooling)







#
