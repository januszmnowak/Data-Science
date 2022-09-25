import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import mglearn

#loading and arranging dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']

#scaler choice and fitting - StandardScaler
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X)

#using scaler to scale data
X_scaled=scaler.transform(X)

#model choice, parametrization and fitting - PCA
from sklearn.decomposition import PCA
model = PCA(n_components=2) #alternatywnie zamiast n_components=2 moge wpisac liczbe z przedzialu (0,1) ktora szuka do ilu kompotentow ma byc redukcja zeby explained variance bylo rowne tej liczbie
model.fit(X_scaled)

#transform data onto principal components
X_pca = model.transform(X_scaled)
print("Model components: {}".format(str(model.components_)))
print("Model explained variance: {}".format(str(model.explained_variance_)))
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second principal component, colored by class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], y)
plt.legend(target_names, loc="best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.show()

#coefficient vizualization
plt.matshow(model.components_, cmap='viridis')
plt.yticks([0, 1], ["First component", "Second component"])
plt.colorbar()
plt.xticks(range(len(feature_names)),feature_names, rotation=60, ha='left')
plt.xlabel("Feature")
plt.ylabel("Principal components")
plt.show()

#chart with explained variance
pca=PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.show()