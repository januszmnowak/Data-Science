import numpy as np

X=np.array([[np.nan,0,3],
            [3,7,9],
            [3,5,2],
            [4,np.nan,6],
            [8,8,1]])

from sklearn.impute import SimpleImputer
imp=SimpleImputer(missing_values=np.nan,strategy='mean')
X2=imp.fit_transform(X)

""" 
Strategy:
- If "mean", then replace missing values using the mean along each column. Can only be used with numeric data.
- If "median", then replace missing values using the median along each column. Can only be used with numeric data.
- If "most_frequent", then replace missing using the most frequent value along each column. Can be used with strings or numeric data.
- If "constant", then replace missing values with fill_value. Can be used with strings or numeric data."""

print(X)
print(X2)