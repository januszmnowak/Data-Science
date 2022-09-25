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

#pipeline and grid search preparation to transform data
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures

#model choice, parametrization and fitting - Linear Regression
from sklearn.linear_model import LinearRegression
#model = LinearRegression() no data preprocessing and no grid search
pipe = Pipeline([("preprocessor",PolynomialFeatures()),("regressor",LinearRegression())])
param_grid={'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)


#model choice, parametrization and fitting - Ridge Regression
"""from sklearn.linear_model import Ridge
#model = Ridge(alpha=1.0) #no data scaling and no grid search
pipe = Pipeline([("preprocessor",PolynomialFeatures()),("regressor",Ridge())])
param_grid={'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__alpha':[0.001,0.01,0.1,1,10,100]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)"""

"""Alpha parameter - default alpha=1.0.
Increasing alpha forces coefficients to move more toward zero, which decreases
training set performance but might help generalization (less complex model, less overfiting)."""

#model choice, parametrization and fitting - Lasso Regression
"""from sklearn.linear_model import Lasso
#model = Lasso(alpha=1.0) #no scaling and no grid search
pipe = Pipeline([("preprocessor",PolynomialFeatures()),("regressor",Lasso())])
param_grid={'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__alpha':[0.001,0.01,0.1,1,10,100]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)"""

#best model
print("Best parameters: {}".format(model.best_params_))
print("Best estimator:\n{}".format(model.best_estimator_))
print("model.coef_: {}".format(model.best_estimator_.steps[-1][1].coef_))
print("model.intercept_: {}".format(model.best_estimator_.steps[-1][1].intercept_))

#testing model accuracy
r2_train=model.score(X_train, y_train)
r2_test=model.score(X_test, y_test)
print("Train accuracy or R2 is: ",r2_train)
print("Test accuracy or R2 is: ",r2_test)

adjr2_train=1-(1-r2_train)*(X_train.shape[0]-1)/(X_train.shape[0]-X_train.shape[1]-1)
adjr2_test=1-(1-r2_test)*(X_test.shape[0]-1)/(X_test.shape[0]-X_test.shape[1]-1)
print("Train Adjusted R2 is: ",adjr2_train)
print("Test Adjusted R2 is: ",adjr2_test)

#using model to classify test data
y_predict=model.predict(X_test)
output=pd.DataFrame(X_test,columns=feature_names).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
print(output)

