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

#pipeline and grid search preparation
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import PolynomialFeatures

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import LinearSVR
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

pipe = Pipeline([("preprocessor",PolynomialFeatures()),("regressor",LinearRegression())])

param_grid=[{'regressor':[LinearRegression()],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2]},{'regressor':[Ridge(solver='lsqr')],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__alpha':[0.001,0.01,0.1,1,10,100]},{'regressor':[Lasso(tol=0.01)],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__alpha':[0.001,0.01,0.1,1,10,100]},{'regressor':[LinearSVR()],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__C':[0.001,0.01,0.1,1,10,100]},{'regressor':[SVR()],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__C':[0.001,0.01,0.1,1,10,100]},{'regressor':[KNeighborsRegressor()],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__n_neighbors':[1,2,3,4,5]}]
model = GridSearchCV(pipe,param_grid,cv=5,scoring="r2")

model.fit(X_train,y_train)

#model best parameters
print("Best parameters: {}".format(model.best_params_))
print("Best estimator:\n{}".format(model.best_estimator_))

#model coefficients
print("model.coef_: {}".format(model.best_estimator_.steps[-1][1].coef_))
print("model.intercept_: {}".format(model.best_estimator_.steps[-1][1].intercept_))

#using model to classify test data
y_predict=model.predict(X_test)
output=pd.DataFrame(X_test).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
print(output)

#testing model accuracy
train_accuracy=model.score(X_train, y_train)
test_accuracy=model.score(X_test, y_test)
print("Train accuracy or R2 is: ",train_accuracy)
print("Test accuracy or R2 is: ",test_accuracy)