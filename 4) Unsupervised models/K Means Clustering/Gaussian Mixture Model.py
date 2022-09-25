import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#data importing
data=pd.read_csv(r'data.csv',sep=';',header=None,skiprows=[0],decimal=',',names=(['netdebt_to_ebitda','debt_to_assets','risk']))
X=data[['netdebt_to_ebitda','debt_to_assets']]
y=data['risk']

#model choice, parametrization and fitting
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
model = GaussianMixture(n_components=4)
model.fit(X)

#using model to cluster data
ynew = model.predict(X)

#printing output
output=pd.DataFrame(X).copy()
output['result']=ynew
print(output)

#printing probabilities
probs = model.predict_proba(X)
print("matrix of size [n_samples, n_clusters] that measures the probability that any point belongs to the given cluster: ",probs.round(3))

#chart
size = 50 * probs.max(1) ** 2
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=ynew,s=size,cmap='rainbow')
plt.show()
