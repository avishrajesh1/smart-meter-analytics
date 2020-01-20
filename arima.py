import pandas as pd
import numpy
#import 
from numpy import nan
import matplotlib
from matplotlib import pyplot

df=pd.read_csv(r"C:\Users\Avish Khosla\OneDrive\Desktop\smart_grid_sop\household_power_consumption.txt",sep=';',header=0,low_memory=False,infer_datetime_format=True, parse_dates={'datetime':[0,1]},index_col=['datetime'])
print(df.head())
df.replace('?',nan,inplace=True)
val=df.values.astype('float32')
df['sub4']=(val[:,0]*1000/60)-(val[:,4]+val[:,5]+val[:,6])
print(df.head())


df=df[['Global_active_power']].astype('float16')
print(df.head())
df=df.resample("180Min").mean()
#df=df.groupby(df.index.min).mean()
#df=df[-5000:]

df.index = pd.to_datetime(df.index)
# arima prediction
print(df)
import plotly.plotly as ply
from pyramid.arima import auto_arima
#warnings.filterwarnings("ignore") # specify to ignore warning messages
import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]
plt.style.use('fivethirtyeight')

for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df.astype('float32'),
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)

            results = mod.fit()

            #print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
            print(9)
        except:
            print(9)
            continue





mod = sm.tsa.statespace.SARIMAX(df,
                                order=(1, 1, 1),
                                seasonal_order=(1, 1, 1, 12),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])
results.plot_diagnostics(figsize=(15, 12))
plt.show()

pred = results.get_prediction(start=10000, dynamic=False)
pred_ci = pred.conf_int()
ax = df.plot(label='observed')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7, figsize=(14, 4))
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
ax.set_xlabel('Date')
ax.set_ylabel('Retail_sold')
plt.legend()
plt.show()
'''
'''
y_forecasted = pred.predicted_mean
y_truth = df['2010-11-21 07:43:00':]
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error is {}'.format(round(mse, 2)))
print('The Root Mean Squared Error is {}'.format(round(np.sqrt(mse), 2)))

'''

import numpy as np
# prepare Fourier terms
y=df
y=y.dropna()
y_to_train = y.iloc[:(len(y)-365)]
y_to_test = y.iloc[(len(y)-365):] 
exog = pd.DataFrame({'date': y.index})
exog = exog.set_index(pd.PeriodIndex(exog['date'], freq='3H'))
exog['sin365'] = np.sin(2 * np.pi * exog.index.dayofyear / 365.25)
exog['cos365'] = np.cos(2 * np.pi * exog.index.dayofyear / 365.25)
exog['sin365_2'] = np.sin(4 * np.pi * exog.index.dayofyear / 365.25)
exog['cos365_2'] = np.cos(4 * np.pi * exog.index.dayofyear / 365.25)
exog = exog.drop(columns=['date'])
exog_to_train = exog.iloc[:(len(y)-365)]
exog_to_test = exog.iloc[(len(y)-365):]
# Fit model
arima_exog_model = auto_arima(y=y_to_train, exogenous=exog_to_train, seasonal=True, m=7)
# Forecast
y_arima_exog_forecast = arima_exog_model.predict(n_periods=365, exogenous=exog_to_test)


'''

print(999999999999999)
pred_uc = results.get_forecast(steps=100)
print(999999999999999)
pred_ci = pred_uc.conf_int()
print(999999999999999)
print(df)

ax1 = df.plot(label='observed', figsize=(14, 4))
print(999999999999999)
pred_uc.predicted_mean.plot(ax=ax1, label='Forecast')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
ax1.set_xlabel('Date')
ax1.set_ylabel('Sales')
plt.legend()
plt.show()