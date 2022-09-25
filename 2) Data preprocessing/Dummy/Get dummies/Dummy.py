import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#loading and arranging dataset
data = pd.DataFrame({'x0':[1,2,3,4,5],'x1':[7,3,5,6,2],'y':[12,14,17,22,28],'category':['a','b','a','a','b'], 'classification':['prime','second','second','prime','prime']})
print(data)

#changing columns with categorical values into dummies
data_dummies = pd.get_dummies(data, columns=None, drop_first=True) #zamiast None mogę podać listę kolumn które mają traktowane jako categorical values
#drop_first=True: used to get k-1 dummies out of k categorical levels by removing the first level - used to avoid mulitcolinnearity between features in regression

print(data_dummies)
