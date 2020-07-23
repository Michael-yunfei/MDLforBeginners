# Chapter 2 C1 C2 Code
# 第一章
# @ Michael


import os
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron


# set the working direction
os.getcwd()
os.chdir('/Users/Michael/Documents/MDLforBeginners/Chapter2/Code')

# import iris dataset
iris = datasets.load_iris()
iris.data
iris.target
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df.shape

wine = datasets.load_wine()
wine.data
wine.target
wine.feature_names.append('Wine')
wine_df = pd.DataFrame(np.column_stack((wine.data, wine.target)),
                       columns=wine.feature_names)

fig = px.scatter_3d(wine_df, x='alcohol', y='malic_acid', z='color_intensity',
                    color='Wine')
fig.show()


# generate two datasets
# 1 - a dataset including height and weight for Chinese men
# 2 - a dataset including the same features for Germany women

# read the height and weight, calculate the covariance first
bmi_hw = pd.read_csv('datasetsWeightHeight.csv')
bmi_hw.shape
bmi_hw.head()
bmi_hw_norm = preprocessing.normalize(bmi_hw.iloc[:, 1:3])
bmi_cov = pd.DataFrame(bmi_hw_norm).cov()

bmi_cov2 = [[3, -0.1], [-0.1, 3]]
chinese_men = np.random.multivariate_normal(mean=(170, 66),
                                            cov=bmi_cov2,
                                            size=100)
german_women = np.random.multivariate_normal(mean=(170, 67),
                                             cov=bmi_cov2,
                                             size=100)
german_women_label = np.column_stack([german_women,
                                      np.zeros((100, 1))])
chinese_men_label = np.hstack([chinese_men,
                               np.ones([100, 1])])

women_men_df = pd.DataFrame(np.vstack([chinese_men_label,
                                       german_women_label]),
                            columns=['height', 'weight', 'gender'])

women_men_df.gender = women_men_df.gender.astype(int)

# plot the dataset
fig, axes = plt.subplots(1, 2, sharey=True, figsize=(12, 6))
sns.scatterplot(x=women_men_df.height,
                y=women_men_df.weight,
                ax=axes[0], s=69)
sns.scatterplot(x=women_men_df.height,
                y=women_men_df.weight,
                hue=women_men_df.gender,
                ax=axes[1], s=69)
fig.savefig('BMI.png', dpi=900, bbox_inches='tight')


# 3D plot for visualizing the seperating plane
women_men_df2 = women_men_df.copy()
women_men_df2.gender = women_men_df2.gender.astype(str)
bmi_plotly_fig = px.scatter_3d(women_men_df2, x='height',
                               y='weight',
                               z='gender', color='gender')
bmi_plotly_fig.show()

# working with iris dataset again
iris = datasets.load_iris()
iris_df2 = pd.DataFrame(np.column_stack([iris.data, iris.target]),
                        columns=iris.feature_names + ['category'])
iris_df2.category = iris_df2.category.astype(int)
iris_df2.head()
iris_df.shape

# build up 3 equations
for i in range(3):
    print(iris_df2[iris_df2['category'] == i].iloc[1, :])

# solve the the linear equations
iris_mat = np.array([[3.0, 1.4, 0.2],
                     [3.2, 4.5, 1.5],
                     [2.7, 5.1, 1.9]])
iris_param = np.linalg.inv(iris_mat) @ np.asmatrix([0, 1, 2]).reshape((3, 1))
print(iris_param)

# 1.1， -3.7, 9.4
iris_df2.iloc[13, :]  # categor 0
iris_df2.iloc[13, 1:4] @ iris_param

# use the parameters from three equations to predict
iris_df2_pred = np.asmatrix(iris_df2.iloc[:, 1:4] @ iris_param)
iris_df2_pred_cat = np.ones((iris_df2_pred.shape))*9
iris_df2_pred_cat.astype(int)
# classifying the dataset
for i in range(iris_df2_pred.shape[0]):
    if iris_df2_pred[i, :] <= 0.2:
        iris_df2_pred_cat[i, :] = 0
    elif iris_df2_pred[i, :] <= 1.2:
        iris_df2_pred_cat[i, :] = 1
    elif iris_df2_pred[i, :] <= 2.2:
        iris_df2_pred_cat[i, :] = 2

iris_check = np.sum(iris_df2_pred_cat.astype(int).ravel() == iris_df2.category)
print(iris_check/iris_df2.shape[0])


# Section 2 Linear classification
# relabel the target
iris_binary_target = np.zeros(iris.target.shape)
for i in range(iris.target.shape[0]):
    if iris.target[i] == 0:
        iris_binary_target[i] = 1
    else:
        iris_binary_target[i] = -1

iris_binary_target

np.random.seed(16)  # set random seed
iris_xtrain, iris_xtest, \
 iris_ytrain, iris_ytest = train_test_split(iris.data,
                                            iris_binary_target,
                                            test_size=0.3)
# select four equations
iris_4equations = np.random.randint(105, size=4)
iris_4eq_values = iris_xtrain[iris_4equations, :]
iris_4eq_w = np.linalg.inv(iris_4eq_values) @ \
 iris_ytrain[iris_4equations].reshape((4, 1))

# print out the coefficients
print(iris_4eq_w)

# predict
iris_4eq_pred = iris_xtest @ iris_4eq_w
iris_4eq_pred2 = np.zeros(iris_4eq_pred.shape)
for i in range(iris_4eq_pred.shape[0]):
    if iris_4eq_pred[i] > 0:
        iris_4eq_pred2[i] = 1
    else:
        iris_4eq_pred2[i] = -1

# check the accuracy
np.sum(iris_4eq_pred2.ravel() == iris_ytest)/iris_ytest.shape[0]
# 71.1% percent, quite impressive

# select 40 equations, and calculate coefficients for each 4 equations
# and then calcualte the averate of w
iris_40equations = np.random.randint(105, size=40)
iris_40eq_values = iris_xtrain[iris_40equations, :]
iris_40eq_w = np.linalg.inv(iris_40eq_values.T @ iris_40eq_values) @ \
 iris_40eq_values.T @ iris_ytrain[iris_40equations].reshape((40, 1))

# predict
iris_40eq_pred = iris_xtest @ iris_40eq_w
iris_40eq_pred2 = np.zeros(iris_40eq_pred.shape)
for i in range(iris_40eq_pred.shape[0]):
    if iris_40eq_pred[i] > 0:
        iris_40eq_pred2[i] = 1
    else:
        iris_40eq_pred2[i] = -1

# check the accuracy
np.sum(iris_40eq_pred2.ravel() == iris_ytest)/iris_ytest.shape[0]
# 100 %, very surpised

# Perceptron Method
iris_percep = Perceptron(tol=1e-3)
iris_percep.fit(iris_xtrain, iris_ytrain)
iris_percept_pred = iris_percep.predict(iris_xtest)
np.sum(iris_percept_pred == iris_ytest)/iris_ytest.shape[0]
# 100% too, dataset is very small.


# show the convergence
# f(x) = -x^2 + 2x + 3
# f'(x) = -2x + 2
def conv_fx(x):
    y = -2*x + 2

    return y


fw = 0
fw_values1 = [0]
fx_values1 = [2]

for i in range(20):
    fw = fw + 0.1 * conv_fx(fw)  # here it is maximizing, so it is
    fw_values1.append(fw)
    fx_values1.append(conv_fx(fw))


fw = 3
fw_values2 = [3]
fx_values2 = [-4]
for i in range(20):
    fw = fw + 0.1 * conv_fx(fw)  # here it is maximizing, so it is
    fw_values2.append(fw)
    fx_values2.append(conv_fx(fw))

conv_d = {'weight1': fw_values1, 'derivative1': fx_values1,
          'weight2': fw_values2, 'derivatives2': fx_values2}
conv_df = pd.DataFrame(conv_d)
print(conv_df.to_latex(index=False))


# show the convergence again
# f(x) = x^2 - 2x - 3
# f'(x) = 2x-2
def conv_fx2(x):
    y = 2*x - 2

    return y


fw = 0
fw_values3 = [0]
fx_values3 = [2]

for i in range(20):
    fw = fw - 0.1 * conv_fx2(fw)  # here it is maximizing, so it is
    fw_values3.append(fw)
    fx_values3.append(conv_fx(fw))

fw = 3
fw_values4 = [3]
fx_values4 = [-4]
for i in range(20):
    fw = fw - 0.1 * conv_fx2(fw)  # here it is maximizing, so it is
    fw_values4.append(fw)
    fx_values4.append(conv_fx(fw))

conv_d2 = {'weight1': fw_values3, 'derivative1': fx_values3,
           'weight2': fw_values4, 'derivatives2': fx_values4}
conv_df2 = pd.DataFrame(conv_d2)
print(conv_df2.to_latex(index=False))








#
