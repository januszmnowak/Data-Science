import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
data_train=pd.read_csv('train.csv')
data_test=pd.read_csv('test.csv')

data_train['Admitted'] = data_train['Admitted'].map({'Yes': 1, 'No': 0})
data_train['Gender'] = data_train['Gender'].map({'Female': 1, 'Male': 0})
data_test['Admitted'] = data_test['Admitted'].map({'Yes': 1, 'No': 0})
data_test['Gender'] = data_test['Gender'].map({'Female': 1, 'Male': 0})

X_train=data_train[['SAT','Gender']]
y_train=data_train['Admitted']
X_test=data_test[['SAT','Gender']]
y_test=data_test['Admitted']

#model choice, parametrization and fitting - Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(C=1.0)
model.fit(X_train,y_train)

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
print("Classification report: \n",classification_report(y_test,y_predict))

#precision-recall curve
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresholds = precision_recall_curve(y_test,model.decision_function(X_test))
close_zero=np.argmin(np.abs(thresholds))

plt.plot(precision,recall,label="precision-recall curve")
plt.plot(precision[close_zero],recall[close_zero],'o',markersize=10,fillstyle="none",c='k',mew=2,label="threshold zero")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc=3)
plt.show()

print("Average precision score: ",average_precision_score(y_test, model.decision_function(X_test)))

#ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test,model.decision_function(X_test))

plt.plot(fpr,tpr,label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.legend(loc=4)
plt.show()

print("ROC AUC score: ",roc_auc_score(y_test, model.decision_function(X_test)))