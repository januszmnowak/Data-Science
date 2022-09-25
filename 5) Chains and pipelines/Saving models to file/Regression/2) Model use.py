import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import pickle

#loading model from pickle
model=pickle.load(open('finalized_model.sav','rb'))

#using model to classify new data
Xnew=np.array([1.5,3.5,5.5,7.5,9.5])
Xnew=Xnew.reshape(-1,1) #zmieniam wymiar
ypredict=model.predict(Xnew)

output=pd.DataFrame(Xnew).copy()
output['result']=ypredict
print(output)








