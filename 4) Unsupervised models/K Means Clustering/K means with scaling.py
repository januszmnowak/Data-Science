import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading, arranging and printing dataset
X=pd.read_csv('data.csv')

#scaler choice and fitting - K means requires scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(X)

#model choice, parametrization and fitting
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(X_scaled)

#using model to cluster data
y_predict=model.predict(X_scaled)

#printing output
output=pd.DataFrame(X).copy()
output['Predicted result']=y_predict
print(output)

#optimum number of clusters - looking for max silhouette score
from sklearn.metrics import silhouette_score
print("Silhouette score (from -1 to +1, higher is better) is: ",silhouette_score(X, model.labels_))

#optimum number of clusters - finding elbow using WCSS (within-cluster sum of squares)
wcss=[]
for i in range(1,11):
    kmeans = KMeans(i)
    kmeans.fit(X_scaled)
    wcss_iter = kmeans.inertia_
    wcss.append(wcss_iter)

number_clusters = range(1,11)
plt.plot(number_clusters,wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Within-cluster Sum of Squares')
plt.show()


# plot dataset
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_predict,s=50,cmap='rainbow')
centers = scaler.inverse_transform(model.cluster_centers_)
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)
plt.show()