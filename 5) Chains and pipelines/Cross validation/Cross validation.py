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

#model choice and parametrization - Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression() #nie musze uzywac metody .fit(X,y)

#cross validation - basic
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y,cv=5,scoring="accuracy") #cv - number of folds
#scoring=? print(sorted(sklearn.metrics.SCORERS.keys()))

#cross validation - KFold
"""from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
kfold = KFold(n_splits=5)
scores = cross_val_score(model, X, y,cv=kfold)"""

#printing score
print("Cross-validation scores: {}".format(scores))
print("Average cross-validation score: {:.2f}".format(scores.mean()))

