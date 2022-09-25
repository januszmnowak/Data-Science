import numpy as np
import pandas as pd

#importing excel file with employees
employees=pd.read_excel('Input/employees_file.xlsx',header=[0])

#importing excel file with salaries
salaries=pd.read_excel('Input/salary_file.xlsx',header=None)
salaries.columns=["name","salary","control_no"]

#importing csv file with supervisors
supervisors=pd.read_csv('Input/supervisors_file.csv',skiprows=[0,1],sep='|',header=None)[:-1] #excluding last row
supervisors.columns=["dep_name","id","supervisor"]

#filling missing data in salaries
median_salary=salaries["salary"].median()
salaries["salary"].fillna(value=median_salary,inplace=True)

#merging employees and salaries files
data=pd.merge(employees,salaries,how="outer")

#adding supervisors
data=pd.merge(data,supervisors,how="left",left_on="department",right_on="dep_name")

#dropping unused columns
data.drop(["dep_name","control_no","id"],axis=1,inplace=True)

#gruping - calculating average salary by department
grouped=data.groupby("department")
avg_salary=grouped["salary"].mean()

#adding average salary by department to each employee
data=pd.merge(data,avg_salary,how="left",on="department",suffixes=('','_avg_department'))

#adding new column based on other columns
data["%_of_avg_dep_salary"]=data["salary"]/data["salary_avg_department"]

#adding new column using function
def level(x):
    if x<5:
        return "entry"
    elif x<10:
        return "mid"
    else:
        return "senior"

data['level']=data['experience'].apply(level)

#data sorting in place
data.sort_values(by='salary',ascending=False,inplace=True)

#data printing with chosen columns only
print(data.iloc[:,[0,2,3,4,7,8]])
print(data.describe())

#pivot table
pivot=pd.pivot_table(data,index="department", columns="level", values="salary",aggfunc='mean', margins=True)
print(pivot)
print("Mid IT salary is: ",pivot.loc["IT","mid"])

#cross table
crosstab=pd.crosstab(data['department'],data['level'], margins=True) #used to calcutate group counts (same as pivot with aggfunc='count')
print(crosstab)

#data output to excel
data.to_excel("output.xlsx")

#printing people who have experiece >=5 and %_of_avg_dep_salary <1
to_be_promoted=data[(data["experience"]>=5) & (data["%_of_avg_dep_salary"]<1)] # & = and, | = or, ~ = not
print(to_be_promoted[["name","experience","%_of_avg_dep_salary"]])