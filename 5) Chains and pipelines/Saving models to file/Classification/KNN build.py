import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading, arranging and printing dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#pipeline and grid search preparation to scale data (KNN requires scaled data)
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler

#model choice, parametrization and fitting
from sklearn.neighbors import KNeighborsClassifier
#model=KNeighborsClassifier() #no data scaling and no grid search
pipe = Pipeline([("scaler",MinMaxScaler()),("classifier",KNeighborsClassifier())])
param_grid={'scaler':[MinMaxScaler(),None],'classifier__n_neighbors':[1,2,3,4,5]}
model = GridSearchCV(pipe,param_grid,cv=5)
model.fit(X_train,y_train)

#model best parameters
print("Best parameters: {}".format(model.best_params_))
print("Best estimator:\n{}".format(model.best_estimator_))

#using model to classify test data
y_predict=model.predict(X_test)
output=pd.DataFrame(X_test).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
output['Correct prediction?']=(y_test==y_predict)
print(output)

#testing model accuracy
train_accuracy=model.score(X_train, y_train)
test_accuracy=model.score(X_test, y_test)
print("Train accuracy is: ",train_accuracy)
print("Test accuracy is: ",test_accuracy)

#confusion matrix
from sklearn.metrics import confusion_matrix
matrix=confusion_matrix(y_test,y_predict)
sns.heatmap(matrix,square=True,annot=True,cbar=False)
plt.xlabel('predicted value')
plt.ylabel('true value')
plt.show()

#precision and recall
from sklearn.metrics import precision_score, recall_score, f1_score
print("Precision score: ",precision_score(y_test,y_predict))
print("Recall score: ",recall_score(y_test,y_predict))
print("f1 score: ",f1_score(y_test,y_predict))

#classification report
from sklearn.metrics import classification_report
print("Classification report: \n",classification_report(y_test,y_predict,target_names=target_names))


import pickle
#saving model to pickle
pickle.dump(model,open('finalized_model.sav','wb'))