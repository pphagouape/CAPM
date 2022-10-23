import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.expected_returns import mean_historical_return
from pypfopt.risk_models import CovarianceShrinkage

##Obteniendo los datos

activos = ["GGAL", "VIST","LOMA","TX"]
fechaInicio = "2020-01-01"
hoy = datetime.today().strftime('%Y-%m-%d')
df_precios = pd.DataFrame()

def datosYahoo(dataframe,nombresActivos,inicio,fin):
    for i in nombresActivos:
        dataframe[i] = data.DataReader(i,data_source='yahoo',start=inicio , end=fin)["Adj Close"]
    return dataframe
df = datosYahoo(df_precios,activos,fechaInicio,hoy)
df


df = df.pct_change().dropna()
#Creamos matriz de pesos para cada activo ("Equally weighted")
weights = []
n_assets = len(activos)

#Equally weighted
for i in range(n_assets):
    weights.append(1/n_assets)

w = np.array(weights)

r = np.array(np.mean(df))
C = np.cov(df.transpose())

#Validamos
print("Rendimiento esperado:", r)
print("Pesos activos:", w)
print("Matriz VarCov:", C)

def mu(w,r):
    '''Rendimiento portafolio anualizado'''
    return sum(w * r * 252)

def sigma(w):
    C = np.cov(df.transpose())
    '''Desv STD portadolio anualizada'''
    return np.dot(w,np.dot(C,w.T)) * 252


def sharpe(w):
    '''Sharpe ratio con rf de 4%'''
    rf = .04
    return (mu(w,r) - rf) / sigma(w)


def neg_sharpe(w):
    '''Sharpe ratio negativo'''
    return -sharpe(w)


def random_ports(n):
    '''Portafolio aleatorios'''
    means, stds = [],[]
    for i in range(n):
        rand_w = np.random.rand(len(activos))
        rand_w = rand_w / sum(rand_w)
        means.append(mu(rand_w, r))
        stds.append(sigma(rand_w))

    return means, stds

print("Sharpe port equal w:", round(sharpe(w),2))
###
import scipy.optimize as optimize

def apply_sum_constraint(inputs):
    total = 1 - np.sum(inputs)
    return total

my_constraints = ({'type': 'eq', "fun": apply_sum_constraint })

###Max Sharpe

result = optimize.minimize(neg_sharpe,
                      w,
                      method='SLSQP',
                      bounds=((0, 1.0), (0, 1.0), (0, 1.0),(0, 1.0)),
                      options={'disp': True},
                      constraints=my_constraints)
###Min Var
result_min_var = optimize.minimize(sigma,
                      w,
                      method='SLSQP',
                      bounds=((0, 1.0), (0, 1.0), (0, 1.0),(0, 1.0)),
                      options={'disp': True},
                      constraints=my_constraints)
print(result)
optimal_w = result["x"]
print(f'Fue exitosa : {result.success}')
print(f'Resultado   : [' + ', '.join(f'{w:.2%}' for w in result_min_var.x) + ']')
#Grafiquemos
n_portfolios = 10000
means, stds = random_ports(n_portfolios)

best_mu = mu(optimal_w, r)
best_sigma = sigma(optimal_w)
best_sharpe = sharpe(optimal_w)
plt.plot(stds, means, 'o', markersize=1)
plt.plot(best_sigma, best_mu, 'x',  markersize=10)
plt.xlabel('std')
plt.ylabel('mean')
plt.title('Mean and standard deviation of returns of randomly generated portfolios')