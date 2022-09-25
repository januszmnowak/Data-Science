import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#loading, arranging and printing dataset
X, y = mglearn.datasets.make_forge()

data=pd.DataFrame(X,columns=['First feature', 'Second feature'])
data['outcome']=y
print(data)

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

# plot dataset
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

#histogram
data.hist(bins=20,figsize=(10,10))
plt.show()

#chart presenting the relationship between features
dataframe=pd.DataFrame(X_train)
grr = pd.plotting.scatter_matrix(dataframe, c=y_train, figsize=(15, 15), marker='o',hist_kwds={'bins': 20}, s=60, alpha=.8)
plt.show()