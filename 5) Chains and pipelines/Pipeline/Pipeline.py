import numpy as np
import matplotlib.pyplot as plt

#data creation
x=np.array([1,2,3,4,5])
y=np.array([4,2,1,3,7])
X=x[:,np.newaxis] #LinearRegression requires 2D array (not 1D array)

#importing models to be used in the pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

#using pipeline to create model - long version (used together with Grid Search)
from sklearn.pipeline import Pipeline
model=Pipeline([("polynomial",PolynomialFeatures(degree=2)),("regression",LinearRegression())])

"""#using pipeline to create model - short version
from sklearn.pipeline import make_pipeline
model=make_pipeline(PolynomialFeatures(degree=2),LinearRegression())"""

#fitting the model
model.fit(X,y)

#making predictions
y_predict=model.predict(X)

#chart
plt.scatter(x,y)
plt.plot(x,y_predict)
plt.show()
