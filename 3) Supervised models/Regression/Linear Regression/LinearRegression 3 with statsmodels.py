import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import statsmodels.api as sm

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
X_train_c = sm.add_constant(X_train) #dodaje zmienna pod intercept bo standardowo OLS szacuje model bez intercept
X_test_c = sm.add_constant(X_test)
model = sm.OLS(y_train, X_train_c).fit()
print(model.summary())

#using model to classify test data
y_predict=model.predict(X_test_c)
output=pd.DataFrame(X_test,columns=feature_names).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
print(output)