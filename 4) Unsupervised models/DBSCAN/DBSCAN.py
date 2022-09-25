import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mglearn

#loading, arranging and printing dataset
X, y = mglearn.datasets.make_forge()

#model choice, parametrization and fitting
from sklearn.cluster import DBSCAN
model = DBSCAN(min_samples=3,eps=1)
model.fit(X)

#using model to cluster data
y_predict=model.fit_predict(X)
output=pd.DataFrame(X).copy()
output['Predicted result']=y_predict
print(output)

# plot dataset
plt.scatter(X[:,0], X[:,1], c=y_predict,s=50,cmap='rainbow')
plt.show()
