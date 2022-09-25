import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1,as_frame= False) #The API of fetch_openml changed between versions. In earlier versions, it returns a numpy.ndarray array. Since 0.24.0 (December 2020), as_frame argument of fetch_openml is set to auto (instead of False as default option earlier) which gives you a pandas.DataFrame for the MNIST data. You can force the data read as a numpy.ndarray by setting as_frame = False

X=mnist['data']
y=mnist['target']
y = y.astype(np.uint8) #original dataset has y as strings, need to convert strings to integers
feature_names=mnist['feature_names']

data=pd.DataFrame(X,columns=feature_names)
data['output']=y
data.info()
print(data)

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0,test_size=0.25)

print("X_train shape: {}".format(X_train.shape))
print("y_train shape: {}".format(y_train.shape))
print("X_test shape: {}".format(X_test.shape))
print("y_test shape: {}".format(y_test.shape))

#visualization of numbers
some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap = plt.cm.binary, interpolation="nearest")
plt.axis("off")
plt.show()
