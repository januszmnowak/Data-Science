import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy import stats

#loading and arranging dataset
data=pd.read_csv(r'data.csv',sep=';',header=None,skiprows=[0],decimal=',',names=(['netdebt_to_ebitda','debt_to_assets','risk']))

X=data[['netdebt_to_ebitda','debt_to_assets']]
y=data['risk']

#model choice, parametrization and fitting
from sklearn.svm import SVC
model = SVC(kernel='linear', C=1E10)
model.fit(X, y)

#using model to classify new data
Xnew=np.array([(2.5,0.53),(0.45,0.25),(4.8,1.0),(1.1,0.25)])
ynew=np.array([2,1,2,2])
ypredict=model.predict(Xnew)

output=pd.DataFrame(Xnew).copy()
output['result']=ypredict
print(output)

#support vector coordinates
print("Support vectors: ",model.support_vectors_)

#chart function
def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a two-dimensional SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',levels=[-1, 0, 1], alpha=0.5,linestyles=['--', '-', '--'])
    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],model.support_vectors_[:, 1],s=300, linewidth=1, facecolors='none');
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

#scatter
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, s=50, cmap='rainbow')
plot_svc_decision_function(model)
plt.scatter(Xnew[:,0], Xnew[:,1], c=ypredict, s=30, cmap=('rainbow'), alpha=0.3) #indeksuje bez loc/iloc bo xnew to Numpy Array
plt.show()

#testing model accuracy
accuracy=model.score(Xnew, ynew)
accuracy2=np.mean((ypredict == ynew))
print("Accuracy is: ",accuracy)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(ynew,ypredict)
sns.heatmap(matrix,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

#classification report
from sklearn.metrics import classification_report
print("Classification report: \n",classification_report(ynew,ypredict))