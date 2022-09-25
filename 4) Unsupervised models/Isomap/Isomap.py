import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#loading and arranging dataset
from sklearn.datasets import load_digits
digits=load_digits()
X=digits['data']
y=digits['target']
print("Original dimension of dataset: ",X.shape)

#visualization of data
fig,axes=plt.subplots(10,10,figsize=(8,8),subplot_kw={'xticks':[],'yticks':[]},gridspec_kw=dict(hspace=0.1,wspace=0.1))

for i,ax in enumerate(axes.flat):
    ax.imshow(digits.images[i],cmap='binary',interpolation='nearest')
    ax.text(0.05,0.05,str(y[i]),transform=ax.transAxes,color='green')
plt.show()   

#model choice, parametrization and fitting - Isomap
from sklearn.manifold import Isomap
model=Isomap(n_components=2)
model.fit(X)

#transform data into reduced dimensionality
X_transformed=model.transform(X)
print("Reduced dimension of dataset: ",X_transformed.shape)

#plotting data with reduced dimensionality
plt.scatter(X_transformed[:,0],X_transformed[:,1],c=y,edgecolor='none',alpha=0.5,cmap='rainbow')
plt.colorbar(label='digit label',ticks=range(10))
plt.clim(-0.5,9.5)
plt.show()

#classification using reduced dimensionality
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
from sklearn.naive_bayes import GaussianNB
model=GaussianNB()
model.fit(X_train,y_train)
y_predict=model.predict(X_test)

#accuracy score
from sklearn.metrics import accuracy_score
print("Accuracy score: ",accuracy_score(y_test,y_predict))

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_predict)
sns.heatmap(matrix,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')