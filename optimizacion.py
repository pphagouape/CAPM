"""
Clase 6: Utilizando Optimizadores
"""
# %% Imports
import scipy.optimize as sco
import numpy as np


# %% Calculando la IRR de un cash-flow usando minimización cuadrática
# El cf pertence a un bono bullet a 3 años con cupón del 5% y pagos semestrales
VALUE = 100.
CF = [(0.5,   2.5),
      (1.0,   2.5),
      (1.5,   2.5),
      (2.0,   2.5),
      (2.5,   2.5),
      (3.0, 102.5)]

# Esta función será el objetivo de la optimización pero, como puede tomar
# valores negativos, la minimización directa no funcionaría ya que iría
# a -infinito. least_squares resuelve esto dado que antes de minimizar la
# eleva al cuadrado.
def npv(irr):
    return VALUE - sum(cf / (1 + irr) ** t for t, cf in CF)

res1 = sco.least_squares(npv, .1)

print(f'Fue exitosa : {res1.success}')
print(f'Resultado   : {res1.x[0]:.2%}\n')


# %% Calculando un portfolio de mínima varianza sujeto a restricciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pandas_datareader import data
from pypfopt import expected_returns
from pypfopt.efficient_frontier import EfficientFrontier
import seaborn as sns
import scipy.optimize as sco
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

rets = np.array(weights)

r = np.array(np.mean(df))
vcm = np.cov(df.transpose())

##Los retornos



# Esta función será el objetivo de la optimización. En este caso, dado que
# la varianza es una forma cuadrática, la minimización funciona sin
# inconvenientes.
def p_var(w):
    return np.dot(np.dot(w, vcm), w)

# Límites min y max para los weights (0 y 1 para asumir 'no leverage')
limites = sco.Bounds([0.] * 4, [1.] * 4)

# Restricciones a la minimización (en formato de producto vectorial con
# vector de weights):
# 1 - Suma de weights = 1
# 2 - Retorno del portfolio = 20% (como ejemplo)
restric = sco.LinearConstraint([[1.] * 4, rets], [1., .12], [1., .12])

# Configuraciones para que la minimización converja OK.
configs = {'method': 'trust-constr', 'jac': '2-point', 'hess': sco.BFGS()}

res2 = sco.minimize(p_var, [1 / 4] * 4, bounds=limites, constraints=restric,
                    **configs)

print(f'Fue exitosa : {res2.success}')
print(f'Resultado   : [' + ', '.join(f'{w:.2%}' for w in res2.x) + ']')