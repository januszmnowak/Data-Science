import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.datasets import make_moons
x, y = make_moons(200, noise=.05, random_state=0)
plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans
model1 = KMeans(2, random_state=0)
model1.fit(x)
ynew1=model1.predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ynew1,s=50, cmap='viridis');
plt.show()

from sklearn.cluster import SpectralClustering
model2 = SpectralClustering(n_clusters=2,affinity='nearest_neighbors',assign_labels='kmeans')
ynew2=model2.fit_predict(x)
plt.scatter(x[:, 0], x[:, 1], c=ynew2,s=50, cmap='viridis');
plt.show()



