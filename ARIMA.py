# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:38:54 2018

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

'''importing the dataset'''

import statsmodels.api as sm

dataset_train = sm.datasets.sunspots.load_pandas().data
#setting up the series and date as index
training_set = dataset_train.iloc[:, 1:2]
training_set.index = pd.DatetimeIndex(start='1700', end='2009', freq='A')
#visualising the plot
training_set.plot()

#Importing all the required functions
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARMA
from statsmodels.graphics.tsaplots import plot_pacf

#plotting the ACF
plot_acf(training_set,lags =20, alpha = 0.05) 

#extracting ACF values
acf_array = acf(training_set)


# Run the ADF test on the price series and print out the results
'''Augmented dickey fuller test
Null hypothesis : There is a unit root in the time series sample
Alternative Hypothesis : The series doesn't have a unit root., The series is stationary.'''

results = adfuller(training_set.iloc[:,0].values)
print("The p-value of the  adfuller is {}".format(results[1]))

if results[1] <= 0.05:
    print("Reject Null hypothesis, The series in statioary")
else:
    print("Do no Reject Null, The series is not stationary")


# Take first difference of the stock Series
chg_stock = training_set.diff()
chg_stock = chg_stock.dropna()

# Plot the ACF and PACF on the same page
fig, axes = plt.subplots(2,1)

# Plot the ACF
plot_acf(chg_stock, lags=20, ax=axes[0])

# Plot the PACF
plot_pacf(chg_stock, lags=20, ax=axes[1])
plt.show()


results = adfuller(chg_stock.iloc[:,0])
print("The p-value of the  adfuller is {}".format(results[1]))

if results[1] <= 0.05:
    print("Reject Null hypothesis, The series in stationary")
else:
    print("Do no Reject Null, The series is not stationary")


"Use pacf and information crtieria to find a good model: we can use both AIC and BIC "


model = ARMA (training_set, order = (3,0))
res = model.fit()
res.plot_predict()

# Fit the data to an AR(p) for p = 0,...,6 , and save the BIC
BIC = np.zeros(7)
for p in range(7):
    mod = ARMA(training_set.iloc[:,0].values, order=(p,0))
    res = mod.fit()
# Save BIC for AR(p)    
    BIC[p] = res.bic
    
# Plot the BIC as a function of p
plt.plot(range(1,7), BIC[1:7], marker='o')
plt.xlabel('Order of AR Model')
plt.ylabel('Baysian Information Criterion')
plt.show()

print("According to Baysian information criteria ., we can use ARMA(3,0) model")

"""
Note : The model seems to have seasonal variation ., it is better to model them using seasonal models like SARIMAX
"""

''' Final Fitting'''

# Forecasting series using an AR(1) model
from statsmodels.tsa.arima_model import ARIMA
mod = ARIMA(training_set, order=(3,0,0))
res = mod.fit()

# Plot the original series and the forecasted series
fig, ax = plt.subplots()
ax = training_set.loc['1950':].plot(ax=ax)
fig = res.plot_predict(start = '1990', end='2012', ax =ax, dynamic=True,plot_insample=True)
plt.show()

