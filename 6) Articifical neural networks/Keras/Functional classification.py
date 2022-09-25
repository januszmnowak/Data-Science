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
X_test_scaled=X_test / 255


#model creation
input = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Flatten(input_shape=X_train.shape[1:])(input)
hidden2 = keras.layers.Dense(300, activation="relu")(hidden1)
hidden3 = keras.layers.Dense(100, activation="relu")(hidden2)
output = keras.layers.Dense(10,activation="softmax")(hidden3)
model = keras.models.Model(inputs=[input], outputs=[output])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",optimizer="sgd",metrics=["accuracy"])

history=model.fit(X_train_scaled, y_train, epochs=5,validation_split=0.2)
                    
#chart
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1]
plt.show()

#model evalution
print("Model loss & accuracy: ",model.evaluate(X_test_scaled, y_test))

#using model for predition
X_new = X_test_scaled[:5]
y_proba = model.predict(X_new)
y_predict=np.argmax(y_proba,axis=1)

output=pd.DataFrame(y_test[:5],columns=["Actual result"])
output['Predicted result']=y_predict
print(output)
print("Probability of being in given class: ",y_proba.round(2))