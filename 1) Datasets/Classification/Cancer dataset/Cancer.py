import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']

data=pd.DataFrame(X,columns=feature_names)
data['outcome']=y
data.info()
print(data.describe())
print(data)
print("Correlations: \n",data.corr()['outcome'].sort_values(ascending=False))

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

#chart presenting the relationship between features
dataframe=pd.DataFrame(X_train,columns=feature_names)
grr = pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()