import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
data=pd.read_csv(r'data.csv',sep=';',header=None,skiprows=[0],decimal=',',names=(['netdebt_to_ebitda','debt_to_assets','risk']))

X=data[['netdebt_to_ebitda','debt_to_assets']]
y=data['risk']

#model choice, parametrization and fitting
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=10, random_state=0)
model.fit(X, y)

#using model to classify new data
Xnew=np.array([(0.5,0.35),(1.2,0.6),(2.8,0.6),(3.7,0.85)])
ynew=np.array([1,2,3,4])
ypredict=model.predict(Xnew)

output=pd.DataFrame(Xnew,columns=('netdebt_to_ebitda','debt_to_assets')).copy()
output['result']=ypredict
print(output)

#scatter
plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y, s=30, cmap='rainbow')
plt.scatter(Xnew[:,0], Xnew[:,1], c=ypredict, s=70, cmap=('rainbow'), alpha=0.3) #indeksuje bez loc/iloc bo xnew to Numpy Array
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
