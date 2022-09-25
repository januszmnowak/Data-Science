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



#model build and hyperparameters search
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=X_train.shape[1:]):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
        options = {}
    model.add(keras.layers.Dense(1, **options))
    model.compile(loss="mse",metrics=['mae'])
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {"n_hidden": [0, 1, 2, 3],"n_neurons": np.arange(1, 100),"learning_rate": reciprocal(3e-4, 3e-2),}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train_scaled, y_train, epochs=20)

#best model and parameters
print("Best parameters: ",rnd_search_cv.best_params_)
print("Best score: ",rnd_search_cv.best_score_)
model=rnd_search_cv.best_estimator_.model

#using best model to predict
X_new=X_test_scaled[:5]
y_predict = model.predict(X_new)

output=pd.DataFrame(X_new).copy()
output['Actual result']=y_test[:5]
output['Predicted result']=y_predict
print(output)