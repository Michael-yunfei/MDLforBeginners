# Chapter 3 Pre-lecture Exercise
# 第三章课前练习
# @ Michael

# Import essential packages
# 导入必要扩展

import numpy as np

# Create a vector and matrix，创建向量和矩阵

a = np.array([1, 2, 3])
print(a)
type(a)
a.dtype
a.ndim
a.shape

a1 = np.array([4, 5, 6])
a2 = np.array([7, 8, 9])
print(a1+a2)  # 简单的矢量计算

b = np.array([1, 2, 3], [5.6, 7.8, 9.9])  # 错误的输入
b = np.array([[1, 2, 3], [5.6, 7.8, 9.9]])
b.dtype
b.ndim
b.shape  # always use shape to check the dimension
print(b)

b1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
b1.shape
b2.shape
print(b1+b2)

np.ones([3, 3])  # 快速创建3x3的0矩阵
np.zeros([6, 6])  # 快速创建6x6的1矩阵


# C1-Q1 的程序化解决

def comper(n, q):
    # A function for calculating all possible combinations of sequences
    # Input:
    # n - maximial length of sequence
    # q - number of digits, e.g., 2 means binary, 3 means ternary
    # Output: number of all possible sequences we can get given n and q
    # 计算所有可能数组
    # 输入:
    # n - 数组的长度的最大值
    # q - 可选数值，比如2 意味着二进制可选0， 1； 3意味着可选0, 1, 2
    # 输出: 所有可能数组的数量
    t = 0  # initialize a variable to store the result
    # 初始化一个变量t,用来储存结果
    for i in range(n):
        temp = pow(q, i+1)
        t = t + temp
    return(t)


# Test our function, 测试我们的方程
result = comper(3, 2)
print(result)
# Try different parameters
comper(32, 2)  # 8589934590
comper(64, 2)  # 36893488147419103230
comper(32, 4)  # 24595658764946068820
comper(64, 4)  # 453709822561251284617832809909024281940

# HOPE, now you understand why we need quantum computer
# 希望现在你能够理解为什么我们需要量子计算器了
