import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


#importing input file
#input file columns=['company','sector','final','weight']
data=pd.read_excel('input.xlsx')


#sum of weights by sector
weights_sum=data.groupby('sector').sum()['weight']


#mapping weights by sectors to each company
df=pd.merge(data,weights_sum,on='sector',how ='left',suffixes=('_company','_sector'))


#intra-sector weight calculation
df['weight']=df['weight_company']/df['weight_sector']


#weighted average sector score calculation
df['average_score']=df['weight']*df['final']
grouped_score=df.groupby('sector').sum()['average_score']
output=pd.DataFrame(grouped_score)


#mapping average sector score to each company
df=pd.merge(df,output,on='sector',how='left',suffixes=('_company','_sector'))


#standard deviation calculation
df['diff^2']=df['weight']*(df['final']-df['average_score_sector'])**2
grouped_variance=df.groupby('sector').sum()['diff^2']
grouped_deviation=grouped_variance**0.5


#average score to percentile conversion
min=float(output.min())
max=float(output.max())
output['percentile']=(output['average_score']-min)/(max-min)


#percentile to score function
def percentile_to_score(x):
    if x<0.10:
        return -2
    elif x<0.35:
        return -1
    elif x<0.65:
        return 0
    elif x<0.90:
        return 1
    else:
        return 2


#percentile to score mapping
output['final_score']=output['percentile'].apply(percentile_to_score)


#std dev addition
output['std_dev']=grouped_deviation[:]


#output sorting in place
output.sort_values(by='average_score',ascending=False,inplace=True)


#summary printing
print(output)
output.to_excel('output.xlsx',sheet_name='sector_scoring')


#bar chart with final score
plt.barh(output.index,output['final_score'])
plt.title('Sector score')
plt.show()


#bar chart with std_dev
std_dev=output['std_dev'].sort_values(ascending=True)
plt.barh(std_dev.index,std_dev[:])
plt.title('Intra-sector std dev')
plt.show()