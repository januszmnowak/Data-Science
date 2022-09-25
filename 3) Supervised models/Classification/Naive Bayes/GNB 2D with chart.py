import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
data=pd.read_csv(r'data.csv',sep=';',header=None,skiprows=[0],decimal=',',names=(['netdebt_to_ebitda','debt_to_assets','risk']))

X=data[['netdebt_to_ebitda','debt_to_assets']]
y=data['risk']
print(type(X))

#model choice, parametrization and fitting
from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X, y)

#using model to classify new data
Xnew=np.array([(2.5,0.53),(0.45,0.25),(4.8,1.0),(1.1,0.25)])
ynew=np.array([2,1,3,2])
ypredict=model.predict(Xnew)

output=pd.DataFrame(Xnew).copy()
output['result']=ypredict
print(output)

#probability matrix
prob=model.predict_proba(Xnew)
print(prob.round(2)) #The columns give the posterior probabilities of the first, second and third label

#scatter
plt.scatter(X.iloc[:,0],X.iloc[:,1],c=y,s=50,cmap=('rainbow')) #indeksuje za pomoca iloc bo x to DataFrame
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