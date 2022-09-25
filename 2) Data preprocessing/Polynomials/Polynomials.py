import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#loading, arranging and printing dataset
X, y = mglearn.datasets.make_wave(n_samples=40)

#preprocessing to polynomials
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=4, include_bias=False)
poly.fit(X)
X_poly = poly.transform(X)

#model choice, parametrization and fitting
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_poly,y)

#model parameters
print("model.coef_: {}".format(model.coef_))
print("model.intercept_: {}".format(model.intercept_))

#using model to classify test data
y_predict=model.predict(X_poly)
output=pd.DataFrame(X_poly).copy()
output['Actual result']=y
output['Predicted result']=y_predict
print(output)

#testing model accuracy
accuracy=model.score(X_poly, y)
print("Accuracy or R2 is: ",accuracy)

#chart
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
line_poly = poly.transform(line)
plt.plot(line, model.predict(line_poly), label='polynomial linear regression')
plt.plot(X_poly[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
