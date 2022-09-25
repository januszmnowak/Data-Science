import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#loading, arranging and printing dataset
X, y = mglearn.datasets.make_wave(n_samples=40)

data=pd.DataFrame(X,columns=['Feature'])
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
mglearn.discrete_scatter(X[:, 0], y)
plt.xlabel("Feature")
plt.ylabel("Target")
plt.show()