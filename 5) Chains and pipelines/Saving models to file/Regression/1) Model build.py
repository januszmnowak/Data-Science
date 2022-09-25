import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle



X=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X=np.array(X).reshape(-1,1) #zmieniam wymiar
y=[4.5, 6, 7.3, 9.2, 10.5, 11.9, 13.6, 15.1, 16.5, 18]


#model choice, parametrization and fitting
from sklearn.linear_model import LinearRegression
model = LinearRegression(fit_intercept=True)
model.fit(X,y)

#slope and intercept output
print("Model slope: ", model.coef_)
print("Model intercept:", model.intercept_)

#saving model to pickle
pickle.dump(model,open('finalized_model.sav','wb'))







