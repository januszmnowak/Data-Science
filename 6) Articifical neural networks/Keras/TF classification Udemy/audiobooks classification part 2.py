"""You are given data from an Audiobook app. Logically, it relates only to the audio versions of books. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.
The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.
You have a .csv summarizing the data. There are several variables: Customer ID, Book length in mins_avg (average of all purchases), Book length in minutes_sum (sum of all purchases), Price Paid_avg (average of all purchases), Price paid_sum (sum of all purchases), Review (a Boolean variable), Review (out of 10), Total minutes listened, Completion (from 0 to 1), Support requests (number), and Last visited minus purchase date (in days).
So these are the inputs (excluding customer ID, as it is completely arbitrary. It's more like a name, than a number).
The targets are a Boolean variable (so 0, or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information.
The task is simple: create a machine learning algorithm, which is able to predict if a customer will buy again.
This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s."""

import numpy as np
import pandas as pd
import tensorflow as tf

#loading data from npz file
data_train = np.load('Audiobooks_data_train.npz')
X_train=data_train['inputs'].astype(float)
y_train=data_train['targets'].astype(float)

data_val = np.load('Audiobooks_data_val.npz')
X_val=data_val['inputs'].astype(float)
y_val=data_val['targets'].astype(float)

data_test = np.load('Audiobooks_data_test.npz')
X_test=data_test['inputs'].astype(float)
y_test=data_test['targets'].astype(float)

#model creation
input_size = 10
output_size = 2
hidden_layer_size = 50
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 1st hidden layer
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

batch_size = 100
max_epochs = 100
early_stopping = tf.keras.callbacks.EarlyStopping(patience=2) #let's set patience=2, to be a bit tolerant against random validation loss increases
model.fit(X_train, # train inputs
          y_train, # train targets
          batch_size=batch_size, # batch size
          epochs=max_epochs, # epochs that we will train for (assuming early stopping doesn't kick in)
          callbacks=[early_stopping], # early stopping
          validation_data=(X_val, y_val), # validation data
          verbose = 2 # making sure we get enough information about the training process
          ) 

model.summary()

#testing model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print('\nTest loss: {0:.2f}. Test accuracy: {1:.2f}%'.format(test_loss, test_accuracy*100.))

#using model for predition
X_new = X_test[:20]
y_proba = model.predict(X_new)
y_predict=np.argmax(y_proba,axis=1)

output=pd.DataFrame(y_test[:20],columns=["Actual result"])
output['Predicted result']=y_predict
output['Correct prediction?']=(output['Actual result']==output['Predicted result'])
output['Probability of being in class 0']=y_proba[:,0].round(2)
output['Probability of being in class 1']=y_proba[:,1].round(2)
print(output)
output.to_excel('output.xlsx')