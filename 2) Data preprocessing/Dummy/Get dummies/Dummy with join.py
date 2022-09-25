import numpy as np
import pandas as pd

#sample dataset
data = pd.DataFrame({'x0':[1,2,3,4,5],'x1':[7,3,5,6,2],'y':[12,14,17,22,28],'category':['a','b','a','a','b']})
print(data)

#converting column with categorical values into dummies
dummies=pd.get_dummies(data['category'], prefix='category')

#joing oryginal dataset ('data') with created dummies ('dummies')
data_dummies=data.join(dummies)

#dropping unnecessary column
data_dummies.drop('category', axis=1, inplace=True)

print(data_dummies)

