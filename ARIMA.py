# -*- coding: utf-8 -*-
"""
Created on Fri Jul  6 22:38:54 2018

@author: Lenovo
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt

#read data
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

#setting up the series and date as index

training_set = dataset_train.iloc[:, 1:2]
training_set.index = pd.to_datetime(dataset_train.iloc[:,0])

#visualising the plot
training_set.plot()

#Importing all the required functions
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.stattools import adfuller

#plotting the ACF
plot_acf(training_set,lags =20, alpha = 0.05) 

#extracting ACF values
acf_array = acf(training_set)


# Run the ADF test on the price series and print out the results
'''Augmented dickey fuller test
Null hypothesis : There is a unit root in the time series sample
Alternative Hypothesis : The series doesn't have a unit root., The series is stationary.'''

results = adfuller(training_set.Open)
print("The p-value of the  adfuller is {}".format(results[1]))





