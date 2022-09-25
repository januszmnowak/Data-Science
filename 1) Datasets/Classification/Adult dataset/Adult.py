import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
data = pd.read_csv("adult.data", header=None, index_col=False,
names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
'income'])
feature_names=['age', 'workclass', 'fnlwgt', 'education', 'education-num',
'marital-status', 'occupation', 'relationship', 'race', 'gender',
'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']


data = data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation', 'income']]

X=data[['age', 'workclass', 'education', 'gender', 'hours-per-week',
'occupation']]
y=data['income']
data.info()
print(data.describe())
print(data)

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#histogram
data.hist(bins=20,figsize=(10,10))
plt.show()