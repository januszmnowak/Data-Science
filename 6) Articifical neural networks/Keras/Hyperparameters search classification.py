import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


#loading fashon_mnist dataset
fashion_mnist=keras.datasets.fashion_mnist

(X_train,y_train),(X_test,y_test)=fashion_mnist.load_data()


#normalizing by dividing by 255
X_train_scaled = X_train / 255.0
X_test_scaled=X_test / 255.0


#model build and hyperparameters search
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=X_train.shape[1:]):
    model = keras.models.Sequential()
    options = {"input_shape": input_shape}
    model.add(keras.layers.Flatten(input_shape=X_train.shape[1:]))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu", **options))
        options = {}
    model.add(keras.layers.Dense(10,activation="softmax",**options))
    model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])
    return model

keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {"n_hidden": [0, 1, 2, 3],"n_neurons": np.arange(1, 300),"learning_rate": reciprocal(3e-4, 3e-2)}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3)
rnd_search_cv.fit(X_train_scaled, y_train, epochs=5)

#best model and parameters
print("Best parameters: ",rnd_search_cv.best_params_)
print("Best score: ",rnd_search_cv.best_score_)
model=rnd_search_cv.best_estimator_.model

#using model for predition
X_new = X_test_scaled[:5]
y_proba = model.predict(X_new)
y_predict=np.argmax(y_proba,axis=1)

output=pd.DataFrame(y_test[:5],columns=["Actual result"])
output['Predicted result']=y_predict
print(output)
print("Probability of being in given class: ",y_proba.round(2))

