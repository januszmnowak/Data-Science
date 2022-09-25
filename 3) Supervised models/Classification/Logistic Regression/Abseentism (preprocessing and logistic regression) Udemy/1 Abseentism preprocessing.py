import numpy as np
import pandas as pd
pd.options.display.max_columns = None #displaying all columns
pd.options.display.max_rows = None #displaying all columns

#importing raw data
raw_csv_data=pd.read_csv('Absenteeism_data.csv')
df=raw_csv_data.copy() #creating the copy of dataset

#looking at data
#display(df)
df.info()

#dropping unncessecary column (ID)
df.drop(['ID'],axis=1,inplace=True)
df.info()

#creating dummies for categorical data (Reason for Absence)
reason_columns=pd.get_dummies(df['Reason for Absence'],drop_first=True)
print(reason_columns.head())

#grouping Reason for Absence into 4 groups
reason_type_1=reason_columns.loc[:,1:14].max(axis=1)
reason_type_2=reason_columns.loc[:,15:17].max(axis=1)
reason_type_3=reason_columns.loc[:,18:21].max(axis=1)
reason_type_4=reason_columns.loc[:,22:].max(axis=1)

#concating df with 4 new columns (reason_type_x)
df=pd.concat([df, reason_type_1, reason_type_2, reason_type_3, reason_type_4],axis=1)

#changing column names
column_list=df.columns.values
print(column_list) #copy list to new list and manually change names
column_list_new=['Reason for Absence','Date','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Education','Children', 'Pets','Absenteeism Time in Hours', 'Reason_1', 'Reason_2', 'Reason_3', 'Reason_4']
df.columns=column_list_new

#removing columns that duplicates the info (Reason for Absence and dummies have the same info)
df.drop(['Reason for Absence'],axis=1,inplace=True)

#reordering columns
column_list_reordered=['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Date','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Education','Children', 'Pets','Absenteeism Time in Hours']
df=df[column_list_reordered]

#creating checkpoint - as df will be further modified I can use created checkpoint (df_reason_mod) if I need interim data preprocessing status
df_reason_mod=df.copy()

#analysing Date column
print(type(df_reason_mod['Date'][0])) #checking data type in column Date
df_reason_mod['Date']=pd.to_datetime(df_reason_mod['Date'],format='%d/%m/%Y')
print(type(df_reason_mod['Date'][0])) #checking data type in column Date


#extracting month value and adding it to df
list_months=[]
for i in range(df_reason_mod.shape[0]):
    list_months.append(df_reason_mod['Date'][i].month)
df_reason_mod['Month Value']=list_months

#extracting day of week and adding it to df
def date_to_weekday(date_value):
    return date_value.weekday()

df_reason_mod['Day of the Week']=df_reason_mod['Date'].apply(date_to_weekday)


#removing date column
df_reason_mod.drop(['Date'],axis=1,inplace=True)

#reordering columns
column_list=df_reason_mod.columns.values
column_list_reordered=['Reason_1','Reason_2','Reason_3','Reason_4','Month Value','Day of the Week','Transportation Expense','Distance to Work','Age','Daily Work Load Average','Body Mass Index','Education','Children','Pets','Absenteeism Time in Hours']
df_reason_mod=df_reason_mod[column_list_reordered]


#creating checkpoint
df_reason_date_mod = df_reason_mod.copy()

#mapping Education column to binary form
"""Turn the data from the ‘Education’ column into binary data, by mapping the value of 0 to the values of 1, and the value of 1 to
the rest of the values found in this column."""
print(df_reason_date_mod['Education'].value_counts())
df_reason_date_mod['Education'] = df_reason_date_mod['Education'].map({1:0, 2:1, 3:1, 4:1})
print(df_reason_date_mod['Education'].value_counts())

#final checkpoint
df_preprocessed = df_reason_date_mod.copy()
print(df_preprocessed.head(10))

#exporting data to cvs file without index
df_preprocessed.to_csv('Absenteeism_preprocessed.csv',index=False)


