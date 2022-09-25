import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

#loading data and choosing "new" data
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X=cancer['data']
y=cancer['target']
target_names=cancer['target_names']
feature_names=cancer['feature_names']

Xnew=X[:50]
ynew=y[:50]

#loading model from pickle
import pickle
model=pickle.load(open('finalized_model.sav','rb'))

#using model to classify new data
output=pd.DataFrame(Xnew).copy()
output['Actual result']=ynew #normally I will not have this data
y_predict=model.predict(Xnew)
output['Predicted result']=y_predict
output['Correct prediction?']=(ynew==y_predict) #normally I will not have this check
print(output)

