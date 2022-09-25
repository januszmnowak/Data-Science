import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import mglearn

#loading, arranging and printing dataset
X, y = mglearn.datasets.make_wave(n_samples=40)

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#pipeline and grid search preparation to scale data (KNNr requires scaled data)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import PolynomialFeatures

#model choice, parametrization and fitting
from sklearn.neighbors import KNeighborsRegressor
#model=NeighborsRegressor(n_neighbors=3) #no data scaling
pipe = Pipeline([("scaler",MinMaxScaler()),("preprocessor",PolynomialFeatures()),("regressor",KNeighborsRegressor())])
param_grid={'scaler':[MinMaxScaler(),None],'preprocessor':[PolynomialFeatures()],'preprocessor__degree':[1,2,3],'regressor__n_neighbors':[1,2,3,4,5]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)

#model best parameters
print("Best parameters: {}".format(model.best_params_))
print("Best estimator:\n{}".format(model.best_estimator_))

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
