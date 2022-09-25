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

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


#model choice, parametrization and fitting - Linear SVR
from sklearn.svm import LinearSVR
#model=LinearSVR() #no data and no grid search
pipe = Pipeline([("scaler",StandardScaler()),("preprocessor",PolynomialFeatures()),("regressor",LinearSVR())])
param_grid={'scaler':[StandardScaler(),None],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__C':[0.001,0.01,0.1,1,10,100]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)

#model choice, parametrization and fitting - SVR
"""from sklearn.svm import SVR
#model=SVR(kernel="poly",degree=2,C=100,epsilon=0.1) #no data scaling and no grid search
pipe = Pipeline([("scaler",StandardScaler()),("preprocessor",PolynomialFeatures()),("regressor",SVR(kernel='poly',degree=2,epsilon=0.1))])
param_grid={'scaler':[StandardScaler(),None],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2],'regressor__C':[0.001,0.01,0.1,1,10,100]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)"""

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