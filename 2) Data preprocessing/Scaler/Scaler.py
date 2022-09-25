import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']


#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#scaler choice and fitting - MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
scaler.fit(X_train)

#scaler choice and fitting - StandardScaler
"""from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)"""

#using scaler to scale data
X_train_scaled=scaler.transform(X_train)
X_test_scaled=scaler.transform(X_test)

#printing min and max
print("X_train per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
print("X_train per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))
print("X_test per-feature minimum after scaling:\n {}".format(X_test_scaled.min(axis=0)))
print("X_test per-feature maximum after scaling:\n {}".format(X_test_scaled.max(axis=0)))