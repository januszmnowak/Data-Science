import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#loading fashon_mnist dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


#dividing set into train and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(housing.data, housing.target)


#scaling data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)


#model creation
model = keras.models.Sequential()
model.add(keras.layers.Dense(30, activation="relu",input_shape=X_train.shape[1:]))
model.add(keras.layers.Dense(1))

model.compile(loss="mean_squared_error",metrics=['mae'])

history=model.fit(X_train_scaled, y_train, epochs=30,validation_split=0.2)
 
model.summary()
              
#chart
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.show()

#model evalution
print("Model evaluation: ",model.evaluate(X_test_scaled, y_test))

#using model for predition
X_new = X_test_scaled[:5]
y_predict = model.predict(X_new)

output=pd.DataFrame(X_new).copy()
output['Actual result']=y_test[:5]
output['Predicted result']=y_predict
print(output)