import pandas as pd
import quandl
import numpy as np
from sklearn import preprocessing,cross_validation
import math, datetime
import matplotlib.pyplot as plt 
from matplotlib import style 
from sklearn.linear_model import LinearRegression
df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
df['HL'] = (df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
df['PCT'] =  (df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
df = df[['Adj. Close','HL','PCT','Adj. Volume',]]
forecast_col='Adj. Close'
df.fillna(-9999,inplace=True)
forecast_out=int(math.ceil(0.01*len(df)))
df['label'] = df[forecast_col].shift(-forecast_out)
print  (forecast_out)

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]
df.dropna(inplace=True)
Y = np.array(df['label'])

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.2)
clf =  LinearRegression()
clf.fit(X_train,Y_train)
acc=(clf.score(X_test,Y_test))
forecast =  clf.predict(X_lately)
print (forecast,acc,forecast_out)
df  ['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day=86400
next_unix = last_unix + one_day
for i in forecast:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix +=one_day
    df.loc[next_date] =[np.nan for _ in range(len(df.columns)-1)] + [i]
print (df.tail())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('DATE')
plt.ylabel('PRICE')
plt.show()
    
    
    
    
