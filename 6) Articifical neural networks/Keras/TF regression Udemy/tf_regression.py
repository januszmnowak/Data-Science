import numpy as np
import pandas as pd
import tensorflow as tf

#generating data
observations = 20
x1 = np.random.uniform(low=-10, high=10, size=(observations,1))
x2 = np.random.uniform(-10, 10, (observations,1))
generated_inputs = np.column_stack((x1,x2))
noise = np.random.uniform(-1, 1, (observations,1))
generated_targets = 2*x1 - 3*x2 + 5 + noise

#saving data to npz file
np.savez('TF_intro', inputs=generated_inputs, targets=generated_targets)

#load data from npz file
training_data=np.load('TF_intro.npz')
X=training_data['inputs']
y=training_data['targets']

#modelc creation
input_size=2
output_size=1

model=tf.keras.Sequential([tf.keras.layers.Dense(output_size)]) #tf.keras.layers.Dense() takes the inputs provided by the model and calculates the dot product of the inputs and the weights and adds the bias [output=np.dot(inputs,weights)+bias]
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(X,y, epochs=200, verbose=2)

model.summary()

#extracting weights and bias
weights=model.layers[0].get_weights()
print("Weights: ",weights[0],"\n", "Bias: ",weights[1])

#Making predictions
output=pd.DataFrame(X).copy()
output['Actual result']=y

y_predict=model.predict_on_batch(X).round(1)
output['Predicted result']=y_predict

print(output)


