import numpy as np
import pandas as pd
import seaborn as sns

#import data
data = pd.read_csv('Countries.csv', index_col='Country')
data.drop(['Language'],axis=1,inplace=True)

#create dendogram with heatmap
sns.clustermap(data, cmap='mako')
