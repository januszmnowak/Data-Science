import numpy as np
import pandas as pd

raw_data = pd.read_csv('Dummies.csv')
print(raw_data)

data = raw_data.copy()
data['Attendance'] = data['Attendance'].map({'Yes': 1, 'No': 0})
print(data)