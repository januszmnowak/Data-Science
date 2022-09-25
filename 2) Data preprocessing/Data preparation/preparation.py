import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.options.mode.chained_assignment = None #disabling warning

#importing data
data=pd.read_excel('data.xlsx')

#initital exploration and visualization
print(data.head())
print("Data info:")
data.info()
print("Data description: \n",data.describe(include='all'))

#dropping unnecessary columns
data.drop(['nip'],axis=1,inplace=True)

#correlation matrix
print("Data correlation matrix: \n",data.corr())
print(sns.heatmap(data.corr()))
pd.plotting.scatter_matrix(data)

#finding missing values
columns=data.columns
for column in columns:
    print("Missing values in column",column,":\n",data[data[column].isnull()])

#droping / filling missing values (rows) manually
data.drop([20,22], axis=0, inplace=True)
data.loc[7,'name']="nowatorski"

#droping missing values (rows) automatically
data.dropna(axis=0, inplace=True, subset=['age'])

#filling missing values automatically
columns=['salary','experience']
for column in columns:
    mean=data[column].mean()
    data[column].fillna(mean,inplace=True)

#finding and replacing outliers
columns=['age','experience']
for column in columns:  
    median = data[column].median()
    std = data[column].std()
    mask = (data[column] - median).abs() > std
    print("Outliers in column",column,":\n",data[mask])
    data[column][mask] = np.nan
    data[column].fillna(median, inplace=True)
    
#finding and dropping duplicates
duplicates=data[data.duplicated(['id'], keep=False)]
print("Duplicates:\n",duplicates)
data=data.drop_duplicates(subset=['id'],keep='first')

print("Cleaned data:\n", data)
data.to_excel('data_cleaned.xlsx')