import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

#current date and time
now=datetime.now()
print("Current date and time: ", now)
print(now.strftime('%Y-%m-%d')) # %Y: four-digit year, %y: two-digit year, %m: month, %d: day, %H: hour, %M: minute, %S: second, %w weekday
print(now.strftime('%H:%M:%S'))

#converting string to date
value='2011-01-03'
date=datetime.strptime(value, '%Y-%m-%d')
print(date)

#time stamps
date=pd.datetime(2015,7,8)
print(now.year, now.month, now.day)

dates=pd.to_datetime([datetime(2015,7,4), '5th of July 2015', '2015-Jul-6', '07-07-2015','20150708'])
print(dates)

dates2=pd.date_range('2015-07-03','2015-07-20')
print(dates2)

dates3=pd.date_range('2015-07-03', periods=8, freq='H')
print(dates3)


#time periods
periods=dates.to_period('D')
print(periods)

#time delta
timedelta=dates-dates[0]
print(timedelta)

timedelta2=pd.timedelta_range(0, periods=9, freq='2H30T')
print(timedelta2)



#importing csv file
data=pd.read_csv('pkn.csv',sep=',')
data = data.set_index(data['Data']) #przyporzadkowuje jako indeks kolumne z datami
data.drop('Data',axis=1,inplace=True) #usuwam zduplikowana kolumne z datami
#data['Data']=pd.to_datetime(data['Data'], format='%Y-%m-%d')
data.index=pd.to_datetime(data.index)

#uzupelniam dni bez notowan metoda forward fill (ffill)
resampled=data.resample('D').ffill() #jeżeli nie chce używać metody ffill to moge zamiast tego wpisać .asfreq() - wtedy dodane zostaną dni, ale brakujące wartosci będą jako NaN

#obliczam roczne stopy zwrotu
resampled['Annual rate of return']=resampled['Zamkniecie']/resampled['Zamkniecie'].shift(365)-1
print(resampled)

#wybieram tylko ceny zamkniecia z tabeli
pkn=resampled['Zamkniecie'] 
print(pkn)

#srednia kroczaca
movavg=pkn.rolling(270).mean()
pkn.plot(style='-')
movavg.plot(style='--')
plt.legend(['raw','moving average'])
plt.title('Moving average')
plt.show()

#resampling
pkn.plot(style='-')
pkn.resample('A').mean().plot(style=':') #resampluje z danych dziennych na roczne i wyliczam srednie z okresu
pkn.asfreq('A').plot(style='--') #zmieniam czestotliwosc na roczna- wybieram wartosci na koniec okresu
plt.legend(['raw','resample','asfreq'])
plt.title('Resampling')
plt.show()





