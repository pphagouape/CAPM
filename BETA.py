import datetime as dt
from turtle import color
from pandas_datareader import DataReader
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

#################------Beta----#############################

# symbols = [stock, market]
symbols = ['ZM', 'SPY']

# Create a dataframe of historical stock prices
# Enter dates as yyyy-mm-dd or yyyy-m-dd
# The date entered represents the first historical date prices will
#   be returned
# Highly encouraged to leave 'Adj Close' as is
data = yf.download(symbols, '2020-5-1')['Adj Close']

# Convert historical stock prices to daily percent change
price_change = data.pct_change()
print(price_change)

# Deletes row one containing the NaN
df = price_change.dropna()
d = price_change.dropna()

#Outliers
zm_mean = d['ZM'].agg(['mean', 'std'])

fig, ax = plt.subplots(figsize=(10,6))
d['ZM'].plot(label='simple_rtn', legend=True, ax = ax)
plt.axhline(y=zm_mean.loc['mean'], c='r', label='mean')
plt.axhline(y=zm_mean.loc['std'], c='c', linestyle='-.',label='std')
plt.axhline(y=-zm_mean.loc['std'], c='c', linestyle='-.',label='std')
plt.legend(loc='lower right')

#Get Outliers
mu = zm_mean.loc['mean']
sigma = zm_mean.loc['std']


cond = (df['ZM'] > mu + sigma * 100) | (df['ZM'] < mu - sigma * 100)
df['outlier'] = np.where(cond, 1, 0)
df['outlier'].value_counts()

#We found 25 outliers if we set 3 times std as the boundary. 
#We can pick those outliers out and put it into another DataFrame and show it in the graph:
outliers = df.loc[df['outlier'] == 1, ['ZM']]
fig, ax = plt.subplots()
ax.plot(df.index, df.ZM, 
        color='blue', label='Normal')
ax.scatter(outliers.index, outliers.ZM, 
           color='red', label='Anomaly')
ax.set_title("ZM's stock returns")
ax.legend(loc='lower right')
plt.tight_layout()

plt.show()

#-----Winsorization
outlier_cutoff = 0.01
df.pipe(lambda x:x.clip(lower=x.quantile(outlier_cutoff),
                        upper=x.quantile(1-outlier_cutoff),
                        axis=1,
                        inplace=True))
df

fig, ax = plt.subplots()
ax.plot(d.index, d.ZM, 
        color='red', label='Normal')
ax.plot(df.index, df.ZM, 
        color='blue', label='Anomaly_removed')
ax.set_title("stock returns outliers_winsorize returns ZM")
ax.legend(loc='lower right');

fig, ax = plt.subplots()
ax.plot(d.index, d.SPY, 
        color='red', label='Normal')
ax.plot(df.index, df.SPY, 
        color='blue', label='Anomaly_removed')
ax.set_title("stock returns outliers_winsorize returns SPY")
ax.legend(loc='lower right');

# Create arrays for x and y variables in the regression model
(BETA, ALPHA, R, P, SE) = stats.linregress(df['SPY'],df['ZM'])

# Informe de la regresión
print(f'Beta: %4.2f\nR2: %4.2f' %(BETA, R**2))

#-----Eliminando Outliers
filtro = df['outlier'] != 1
df_clean = df[filtro]

df_clean['outlier'].value_counts() #No hay putliers

(BETA, ALPHA, R, P, SE) = stats.linregress(df_clean['SPY'],df_clean['ZM'])

# Informe de la regresión
print(f'Beta sin outliers: %4.2f\nR2: %4.2f' %(BETA, R**2))

#-------------------------BETA CON OUTLIERS-------------------
# Descarga datos de ejemplo
ZM = DataReader('ZM', 'yahoo', start = dt.datetime(2020,1,1),
end = datetime.today().strftime('%Y-%m-%d'))
SP500 = DataReader('^GSPC', 'yahoo', start = dt.datetime(2020,1,1),
end = datetime.today().strftime('%Y-%m-%d'))

# Calcula las rentabilidades diarias
ZM['ret'] = ZM['Adj Close'].pct_change()
SP500['ret'] = SP500['Adj Close'].pct_change()

# Calcula los coeficientes de la regresión
(BETA, ALPHA, R, P, SE) = stats.linregress(SP500.ret.dropna(),
ZM.ret.dropna())

# Informe de la regresión
print(f'Beta: %4.2f\nR2: %4.2f' %(BETA, R**2))

# Gráfico de regresión
plt.scatter(SP500.ret, ZM.ret, alpha=0.3)
plt.plot(SP500.ret, ALPHA + BETA * SP500.ret, color = 'r')
plt.axvline(ls = ':', color = '0.2')
plt.axhline(ls = ':', color = '0.2')
plt.title('Rentabilidades diarias 2011')
plt.xlabel('SP500')
plt.ylabel('ZM')
plt.show()