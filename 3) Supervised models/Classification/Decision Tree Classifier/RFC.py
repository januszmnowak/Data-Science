import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading and arranging dataset
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']

#dividing dataset into train and test samples
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#model choice, parametrization and fitting
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=5, random_state=2)
model.fit(X_train,y_train)

#using model to classify test data
y_predict=model.predict(X_test)
output=pd.DataFrame(X_test,columns=feature_names).copy()
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

#precision-recall curve
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresholds = precision_recall_curve(y_test,model.predict_proba(X_test)[:,1])
close_zero=np.argmin(np.abs(thresholds))

plt.plot(precision,recall,label="precision-recall curve")
plt.plot(precision[close_zero],recall[close_zero],'o',markersize=10,fillstyle="none",c='k',mew=2,label="threshold zero")
plt.xlabel("Precision")
plt.ylabel("Recall")
plt.legend(loc=3)
plt.show()

print("Average precision score: ",average_precision_score(y_test, model.predict_proba(X_test)[:,1]))

#ROC curve
from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test,model.predict_proba(X_test)[:,1])

plt.plot(fpr,tpr,label="ROC Curve")
plt.xlabel("FPR")
plt.ylabel("TPR (recall)")
plt.legend(loc=4)
plt.show()

print("ROC AUC score: ",roc_auc_score(y_test,model.predict_proba(X_test)[:,1]))

#feature importance print
fi=pd.DataFrame(feature_names,columns=['feature'])
fi['importance']=model.feature_importances_
fi.sort_values(by='importance',ascending=False,inplace=True)
print(fi)

#feature importance chart
def plot_feature_importances(model):
    n_features=X.shape[1]
    plt.barh(fi['feature'],fi['importance'], align='center')
    plt.yticks(np.arange(n_features), feature_names)
    plt.xlabel("Feature importance")
    plt.ylabel("Feature")
plot_feature_importances(model)
plt.show()