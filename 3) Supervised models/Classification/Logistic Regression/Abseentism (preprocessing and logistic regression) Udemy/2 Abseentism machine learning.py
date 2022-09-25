import numpy as np
import pandas as pd
pd.options.display.max_columns = None #displaying all columns
pd.options.display.max_rows = None #displaying all columns

#importing preprocessed data
data_preprocessed=pd.read_csv('Absenteeism_preprocessed.csv')

#creating target variable
"""if 'Absenteeism Time in Hours' > median then 1, else 0"""
median=data_preprocessed['Absenteeism Time in Hours'].median()
targets=np.where(data_preprocessed['Absenteeism Time in Hours'] > median, 1, 0)
data_preprocessed['Excessive Absenteeism'] = targets

#checking if dataset is balanced
print("Proportion of absent in total dataset: ",targets.sum() / targets.shape[0])

#removing columns
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours'],axis=1)

#creating X and y
X=data_with_targets.iloc[:,:-1]
y=data_with_targets.iloc[:,-1]

#standarization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled=scaler.transform(X)

#train / test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 20)

#logistic regression
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

model=LogisticRegression()
model.fit(X_train,y_train)

#testing the model on train data
y_train_predict=model.predict(X_train)
accuracy_train_1=model.score(X_train, y_train)
accuracy_train_2=np.sum((y_train_predict==y_train))/ y_train_predict.shape[0]
print("Model train accuracy: ",accuracy_train_2)

#table with coefficients
feature_names = X.columns.values
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_names)
summary_table['Coefficient'] = np.transpose(model.coef_)

summary_table.index = summary_table.index + 1 #moving all indices by 1 to make space for intercept
summary_table.loc[0] = ['Intercept', model.intercept_[0]] #adding the intercept at index 0
summary_table['Odds_ratio'] = np.exp(summary_table['Coefficient'])
summary_table.sort_values('Odds_ratio', ascending=False,inplace=True) #sorting the df by 'Odds_ratio'
print(summary_table)

#testing the model on test data
accuracy_test=model.score(X_test, y_test)
print("Model test accuracy: ",accuracy_test)

#saving the model and scaler
import pickle
pickle.dump(model,open('finalized_model.sav','wb'))
pickle.dump(model,open('scaler.sav','wb'))




