# Chapter 3 C1 Exercise
# 第三章课前练习
# @ Michael

# Import essential packages
# 导入必要扩展

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mtmg

# Discover images (探索图片)
# We will import two images and explore how computer store and display images
# 我们将会读取两张图片来探索计算机是如何存储和显示图片的

# The cute panda 可爱的大熊猫
im_panda = mtmg.imread(
    '/Users/Michael/Documents/MDLforBeginners/Chapter3/Code/Images/panda.JPG')
# read panda image 熊猫图片读取
plt.imshow(im_panda)  # show image, 图片显示

im_panda.shape  # (1400, 1400, 3)
print(type(im_panda))   # <class 'numpy.ndarray'>
print(im_panda[:10, :10, 1])  # print 10 by 10 matrix in the second layer
plt.imshow(im_panda[:, :, 0], cmap='gray')
plt.imshow(im_panda[:, :, 1])
plt.imshow(im_panda[:, :, 2])
print(im_panda[700:710, 700:710, 0])
print(im_panda[700:710, 700:710, 1])
print(im_panda[700:710, 700:710, 2])

fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(15, 9))
for i, ax in zip(range(3), ax):
    split_temp = np.zeros(im_panda.shape, dtype='uint8')
    split_temp[:, :, i] = im_panda[:, :, i]
    ax.imshow(split_temp)


# The Mosaic Painting 马赛克绘画
# prefer jpg format, rather than png format
im_mosaic1 = mtmg.imread(
    '/Users/Michael/Documents/MDLforBeginners/Chapter3/Code/Images/Mosaic1.jpg')
plt.imshow(im_mosaic1)
plt.savefig(
    '/Users/Michael/Documents/MDLforBeginners/Chapter3/Notes/fig/catmosaic.png',
    figsize=(9, 6), dpi=156)
im_mosaic1.shape  # (1034, 1562, 4)
print(im_mosaic1[:10, :10, 1])
plt.imshow(im_mosaic1[:, :, 0])
print(im_mosaic1[700:710, 700:710, 0])
print(im_mosaic1[700:710, 700:710, 1])
print(im_mosaic1[700:710, 700:710, 2])

fig, ax = plt.subplots(1, 3, figsize=(15, 9))
for i, ax in zip(range(3), ax):
    split_temp = np.zeros(im_mosaic1.shape, dtype='uint8')
    split_temp[:, :, i] = im_mosaic1[:, :, i]
    ax.imshow(split_temp)


# The RGB Mosaic
# 三色马赛克
im_rgbmosaic = mtmg.imread(
    '/Users/Michael/Documents/MDLforBeginners/Chapter3/Code/Images/rgbmosaic.jpg')
plt.imshow(im_rgbmosaic)
im_rgbmosaic.shape
plt.savefig(
    '/Users/Michael/Documents/MDLforBeginners/Chapter3/Notes/fig/rgbmosaic.png',
    figsize=(6, 9), dpi=156)


# pin down the red color
print(im_rgbmosaic[500:520, 600:620, 0])
print(im_rgbmosaic[500:520, 600:620, 1])
print(im_rgbmosaic[500:520, 600:620, 2])
plt.imshow(im_rgbmosaic[500:520, 600:620, :])

# pin down the blue color
print(im_rgbmosaic[500:520, 230:250, 0])
print(im_rgbmosaic[500:520, 230:250, 1])
print(im_rgbmosaic[500:520, 230:250, 2])
plt.imshow(im_rgbmosaic[500:520, 230:250, :])

# pin down the green color
print(im_rgbmosaic[800:820, 830:850, 0])
print(im_rgbmosaic[800:820, 830:850, 1])
print(im_rgbmosaic[800:820, 830:850, 2])
plt.imshow(im_rgbmosaic[800:820, 830:850, :])
