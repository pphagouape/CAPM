import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import seaborn as sns
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

##Obteniendo los datos

activos = ["GGAL", "VIST","LOMA","TX","^GSPC"]
fechaInicio = "2020-01-01"
hoy = datetime.today().strftime('%Y-%m-%d')
df_precios = pd.DataFrame()

def datosYahoo(dataframe,nombresActivos,inicio,fin):
    for i in nombresActivos:
        dataframe[i] = data.DataReader(i,data_source='yahoo',start=inicio , end=fin)["Adj Close"]
    return dataframe
df = datosYahoo(df_precios,activos,fechaInicio,hoy)
df


plt.figure(figsize=(12.2,4.5))
for i in df.columns.values:
    plt.plot( df[i],  label=i)
plt.title('Precio de las Acciones')
plt.xlabel('Fecha',fontsize=18)
plt.ylabel('Precio en USD',fontsize=18)
plt.legend(df.columns.values, loc='upper left')
plt.savefig('plotprecios.png', dpi=300, bbox_inches='tight')
plt.show()

##Los retornos

df = np.log(df).diff()
df = df.dropna()
df

##Retornos Normales

plt.figure(figsize=(12.2,4.5))
for i in df.columns.values:
    plt.hist( df[i],  label=i, bins = 200)
plt.title('Histograma de los retornos')
plt.xlabel('Fecha',fontsize=18)
plt.ylabel('Precio en USD',fontsize=18)
plt.legend(df.columns.values)
plt.savefig('plotretornosnormales.png', dpi=300, bbox_inches='tight')
plt.show()

####CAPM

##Separamos lo Retornos de benchmark

df_activos =  df.loc[:, df.columns != '^GSPC']
df_activos

df_benchmark1 =  df.loc[:, df.columns == '^GSPC']


##Retornos segun CAPM
retornos1 = expected_returns.capm_return(df_activos, market_prices = df_benchmark1, returns_data= True, risk_free_rate=0.07/100, frequency=252)
retornos1

####Optimización
##Pesos

def pesosPortafolio(dataframe):
    array = []
    for i in dataframe.columns:
        array.append(1/len(dataframe.columns))
    arrayFinal = np.array(array)
    return arrayFinal
pesos = pesosPortafolio(df)
pesos

##Cov

df_cov = df.cov()*252
df_cov

#Varianza del Portafolio
varianza_portafolio = pesos.T @ df_cov @pesos
"La varianza del portafolio es:" + " " + str(round(varianza_portafolio*100,1))+"%"


volatilidad_portafolio = np.sqrt(varianza_portafolio)
"La volatilidad del portafolio es:" + " " + str(round(volatilidad_portafolio*100,1))+"%"


retorno_portafolio = np.sum(pesos*retornos1)
'El retorno anual del portafolio es:' + ' ' + str(round(retorno_portafolio*100,3)) + '%'


##Optimización para mínima varianza: con venta corta

ef = EfficientFrontier(retornos1, df_cov, weight_bounds=(-1,1))
weights = ef.min_volatility()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

##Optimización para mínima varianza: sin venta corta

ef = EfficientFrontier(retornos1, df_cov, weight_bounds=(0,1))
weights = ef.min_volatility()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


##ROptimización para el Sharpe Ratio: con venta corta.

ef = EfficientFrontier(retornos1, df_cov, weight_bounds=(-1,1))
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)


##Optimización para el Sharpe Ratio: sin venta corta.
ef = EfficientFrontier(retornos1, df_cov,weight_bounds=(0,1))
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()
print(cleaned_weights)
ef.portfolio_performance(verbose=True)

##Matriz de Corr
correlation_mat = df.corr()
plt.figure(figsize=(12.2,4.5))
sns.heatmap(correlation_mat, annot = True)
plt.title('Matriz de Correlación')
plt.xlabel('Activos',fontsize=18)
plt.ylabel('Activos',fontsize=18)
plt.show()

