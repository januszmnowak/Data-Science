import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
data=pd.read_csv(r'data.csv')
print(data.head())

X=data['SAT']
X=X.values.reshape(-1,1) #zmieniam wymiar - jest to niezbędne tylko w przypadku regresji z jedna zmienna
y=data['GPA']

#model choice, parametrization and fitting
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X,y)

#slope and intercept output
print("Model slope: ", model.coef_)
print("Model intercept:", model.intercept_)

#R2 and Adjusted R2
r2=model.score(X,y)
print("Model R2:", r2)
adjr2=1-(1-r2)*(X.shape[0]-1)/(X.shape[0]-X.shape[1]-1) #adjr2 = 1-(1-r2)*(n-1)/(n-p-1) where n=number of observation and p=numbr of features
print("Model Adjusted R2:", adjr2)

#using model to classify new data
X_new = pd.DataFrame(data=[1740,1760,1950],columns=['SAT'])
y_predict=model.predict(X_new)
output=pd.DataFrame(X_new)
output['GPA_predicted']=y_predict
print(output)

#scatter
plt.scatter(X,y)
yhat = model.coef_*X + model.intercept_
plt.plot(X,yhat, lw=4, c='orange', label ='regression line')
plt.xlabel('SAT')
plt.ylabel('GPA')
plt.show()





