"""You are given data from an Audiobook app. Logically, it relates only to the audio versions of books. Each customer in the database has made a purchase at least once, that's why he/she is in the database. We want to create a machine learning algorithm based on our available data that can predict if a customer will buy again from the Audiobook company.
The main idea is that if a customer has a low probability of coming back, there is no reason to spend any money on advertizing to him/her. If we can focus our efforts ONLY on customers that are likely to convert again, we can make great savings. Moreover, this model can identify the most important metrics for a customer to come back again. Identifying new customers creates value and growth opportunities.
You have a .csv summarizing the data. There are several variables: Customer ID, Book length in mins_avg (average of all purchases), Book length in minutes_sum (sum of all purchases), Price Paid_avg (average of all purchases), Price paid_sum (sum of all purchases), Review (a Boolean variable), Review (out of 10), Total minutes listened, Completion (from 0 to 1), Support requests (number), and Last visited minus purchase date (in days).
So these are the inputs (excluding customer ID, as it is completely arbitrary. It's more like a name, than a number).
The targets are a Boolean variable (so 0, or 1). We are taking a period of 2 years in our inputs, and the next 6 months as targets. So, in fact, we are predicting if: based on the last 2 years of activity and engagement, a customer will convert in the next 6 months. 6 months sounds like a reasonable time. If they don't convert after 6 months, chances are they've gone to a competitor or didn't like the Audiobook way of digesting information.
The task is simple: create a machine learning algorithm, which is able to predict if a customer will buy again.
This is a classification problem with two classes: won't buy and will buy, represented by 0s and 1s."""

import numpy as np
import tensorflow as tf

#loading data
raw_csv_data = np.loadtxt('Audiobooks_data.csv',delimiter=',')
X=raw_csv_data[:,1:-1] #The inputs are all columns in the csv, except for the first one and last one
y=raw_csv_data[:,-1] #outputs are in the last column

#shuffling data
shuffled_indices = np.arange(X.shape[0])
np.random.shuffle(shuffled_indices)
X = X[shuffled_indices]
y = y[shuffled_indices]

#balancing the dataset (there are more 0 than 1 as targets)
num_one_targets = int(np.sum(y))
zero_targets_counter = 0
indices_to_remove = []
for i in range(X.shape[0]):
    if y[i] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

X = np.delete(X, indices_to_remove, axis=0)
y = np.delete(y, indices_to_remove, axis=0)

#scaling inputs
from sklearn import preprocessing
X = preprocessing.scale(X)

#shuffling data
"""We shuffle the indices before balancing (to remove any day effects, etc.)
However, we still have to shuffle them AFTER we balance the dataset as otherwise, all targets that are 1s will be contained in the train_targets."""
shuffled_indices = np.arange(X.shape[0])
np.random.shuffle(shuffled_indices)
X = X[shuffled_indices]
y = y[shuffled_indices]

#Split the dataset into train, validation, and test
samples_count = X.shape[0]
train_samples_count = int(0.8 * samples_count)
validation_samples_count = int(0.1 * samples_count)
test_samples_count = samples_count - train_samples_count - validation_samples_count

X_train = X[:train_samples_count]
y_train = y[:train_samples_count]

X_val = X[train_samples_count:train_samples_count+validation_samples_count]
y_val = y[train_samples_count:train_samples_count+validation_samples_count]

X_test = X[train_samples_count+validation_samples_count:]
y_test = y[train_samples_count+validation_samples_count:]

# Print the number of targets that are 1s, the total number of samples, and the proportion for training, validation, and test.
print(np.sum(y_train), train_samples_count, np.sum(y_train) / train_samples_count)
print(np.sum(y_val), validation_samples_count, np.sum(y_val) / validation_samples_count)
print(np.sum(y_test), test_samples_count, np.sum(y_test) / test_samples_count)

#saving dataset in npz file
np.savez('Audiobooks_data_train', inputs=X_train, targets=y_train)
np.savez('Audiobooks_data_val', inputs=X_val, targets=y_val)
np.savez('Audiobooks_data_test', inputs=X_test, targets=y_test)
