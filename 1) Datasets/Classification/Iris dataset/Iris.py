import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
from sklearn.datasets import load_iris
iris_dataset = load_iris()

X=iris_dataset['data']
y=iris_dataset['target']
target_names=iris_dataset['target_names']
feature_names=iris_dataset['feature_names']

data=pd.DataFrame(X,columns=feature_names)
data['gatunek']=y
data.info()
print(data.describe())
print(data)
print("Correlations: \n",data.corr()['gatunek'].sort_values(ascending=False))

#dividing iris dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#histogram
data.hist(bins=20,figsize=(15,15))
plt.show()

#chart presenting the relationship between features
dataframe=pd.DataFrame(X_train,columns=feature_names)
grr = pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)
