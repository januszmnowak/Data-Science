import pandas as pd
import numpy as np

data=pd.read_csv('pf2.PF02',skiprows=[0],sep='|',header=None,names=['no','code','isin','name','currency','volume','price','value','purchase_price','%wab','col9','col10','exchange','market_currency','%nav'],index_col=3,decimal=",")
print(data)


navdata=pd.read_csv('pf15.PF15',skiprows=[0],sep='|',header=None,decimal=",")
nav=float(navdata.iloc[0,1])

data['%nav']=data['value']/nav

output=data[['volume','price','value','%nav']].sort_values(by='name',ascending=False)

output.to_excel(r'C:\Users\Nowak\Desktop\Python\portfolio.xls')



