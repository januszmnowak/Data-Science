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
import statsmodels.api as sm

X_train = sm.add_constant(X_train)
model = sm.Logit(y_train,X_train)
model = model.fit()
model.summary()

#using model to classify test data
X_test = sm.add_constant(X_test)
y_predict=model.predict(X_test) #this gives probability
y_predict = list(map(round, y_predict)) #rounding to 0 or 1 based on probability

output=pd.DataFrame(X_test).copy()
output['Actual result']=y_test
output['Predicted result']=y_predict
output['Correct prediction?']=(y_test==y_predict)
print(output)

#Confusion matrix and accuracy
bins=np.array([0,0.5,1])
cm = np.histogram2d(y_test, y_predict, bins=bins)[0]
accuracy = (cm[0,0]+cm[1,1])/cm.sum()
cm_df= pd.DataFrame(cm)
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
print(cm_df)
print("Accuracy is: ",accuracy)
