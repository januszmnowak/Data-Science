import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
rng = np.random.RandomState(1)
X = np.dot(rng.rand(2, 2), rng.randn(2, 200)).T

#model choice, parametrization and fitting - PCA
from sklearn.decomposition import PCA
model = PCA(n_components=1)
model.fit(X)

#transform data onto principal components
X_pca = model.transform(X)
print("Model components: {}".format(str(model.components_)))
print("Model explained variance: {}".format(str(model.explained_variance_)))
print("Original shape: {}".format(str(X.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

#transform data from principal components to original reduced features
X_new = model.inverse_transform(X_pca)

plt.subplot(2, 1, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
plt.scatter(X_new[:, 0], X_new[:, 1], alpha=0.8)
plt.axis('equal')
plt.show()
