import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading, arranging and printing dataset
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()

X=housing['data']
y=housing['target']
feature_names=housing['feature_names']

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#model choice, parametrization and fitting - Linear Regression
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train,y_train)

#calculating p_values using f_regression - f_regression finds the F-statistics and p_values for the *simple* regressions created with each of the independent variables
from sklearn.feature_selection import f_regression
p_values = f_regression(X_train,y_train)[1]

#model summary
print("Model.intercept_: {}".format(model.intercept_))
model_summary=pd.DataFrame(data=feature_names,columns=['Features'])
model_summary ['Coefficients'] = model.coef_
model_summary ['p-values'] = p_values.round(4)
print(model_summary)

#testing model accuracy
r2_train=model.score(X_train, y_train)
r2_test=model.score(X_test, y_test)
print("Train accuracy or R2 is: ",r2_train)
print("Test accuracy or R2 is: ",r2_test)

adjr2_train=1-(1-r2_train)*(X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)
adjr2_test=1-(1-r2_test)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)
print("Train Adjusted R2 is: ",adjr2_train)
print("Test Adjusted R2 is: ",adjr2_test)
"""
#using model to classify test data
y_predict=model.predict(X_test)
output=pd.DataFrame(X_test,columns=feature_names).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
print(output)"""

