# Chapter 1 C1 C2 Code
# 第一章
# @ Michael


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras


# set the working direction
os.getcwd()
os.chdir('/Users/Michael/Documents/MDLforBeginners/Chapter1/Code')


# define a simple function(定义一个方程)
def fx(x):
    """
    A simple function for calcuating:
        f(x) = x^2 + 2x - 3
    Input: x - 1 by n vector
    Output: return f(x) 1 by n vector
    """

    y = np.power(x, 2) + 2 * x - 3

    return y


# Test the function(测试方程)
x = np.array(range(1, 10))
print(fx(x))


# A simple nueral network model for learning the function fx
# Reference: Laurence Moroney
def Learn_model(y_new):
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=float)
    ys = np.array([0.0, 5.0, 12.0, 21.0, 32.0, 45.0, 60.0, 77.0], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=800)
    return model.predict(y_new)[0]


prediction = Learn_model([9.0])
print(prediction)


def c1c2fun1(x):
    """
    解方程 56x + 8y = 36
    输入: x
    输出: y
    """
    y = (36-56*x)/8

    return y


print(c1c2fun1(np.array(range(0, 8))))
# [ 4.5 -2.5  -9.5 -16.5 -23.5 -30.5 -37.5 -44.5]


# Linear regression for predicting house price
# read the dataset
tb_real_estate = pd.read_excel('Real estate valuation data set.xlsx')
tb_real_estate.shape
tb_real_estate.head()
tb_real_estate.columns
# Index(['No', 'X1 transaction date', 'X2 house age',
#        'X3 distance to the nearest MRT station',
#        'X4 number of convenience stores', 'X5 latitude', 'X6 longitude',
#        'Y house price of unit area'],
#       dtype='object')

# Data visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey=True)  # share y
sns.scatterplot(x=tb_real_estate['X2 house age'],
                y=tb_real_estate.iloc[:, 7],
                ax=axes[0, 0])
sns.scatterplot(x=tb_real_estate.iloc[:, 3],
                y=tb_real_estate.iloc[:, 7], ax=axes[0, 1])
sns.scatterplot(x=tb_real_estate.iloc[:, 4],
                y=tb_real_estate.iloc[:, 7], ax=axes[1, 0])
sns.scatterplot(x=tb_real_estate.iloc[:, 5],
                y=tb_real_estate.iloc[:, 7], ax=axes[1, 1])
fig.savefig('taibeiRealEst.png', dpi=800, bbox_inches='tight')

# run regression
tb_X = tb_real_estate.iloc[:, 2:7]
tb_Y = tb_real_estate.iloc[:, 7:8]

tb_reg = LinearRegression()
tb_reg.fit(tb_X, tb_Y)
tb_reg.coef_
# array([[-2.68916833e-01, -4.25908898e-03,  1.16302048e+00,
#          2.37767191e+02, -7.80545273e+00]])
tb_reg.intercept_

# notice that it does not make sense for normalizing the distance!
tb_x_norm = preprocessing.normalize(tb_X)
tb_norm_reg = LinearRegression()
tb_norm_reg.fit(tb_x_norm, tb_Y)
tb_norm_reg.coef_
# array([[-6.71038947e+01,  7.82866997e+01,  1.79603302e+02,
#          8.48265938e+04, -1.73503097e+04]])
# the second coefficient should be negative
#
