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
# ['alcohol',
#  'malic_acid',
#  'ash',
#  'alcalinity_of_ash',
#  'magnesium',
#  'total_phenols',
#  'flavanoids',
#  'nonflavanoid_phenols',
#  'proanthocyanins',
#  'color_intensity',
#  'hue',
#  'od280/od315_of_diluted_wines',
#  'proline', 'Wine']


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















#
