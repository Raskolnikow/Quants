import numpy as np
import pandas as pd
import pandas.io.data as web

import matplotlib as mpl
import matplotlib.pyplot as plt

def f(x):
	"""
		Das ist die allererste Funktion

		x: input as ndarray
	"""
	return np.sin(x)


def loaddata(stock_name):
	"""

	"""

	quotes = web.DataReader(stock_name, data_source='yahoo', start='1/1/2000', end='4/14/2014')
	return quotes 


# ----------------------------------------------

sp500 = loaddata('^GSPC')
sp500.info()

# sp500['Close'].plot(grid=True, figsize=(8, 5))

# Calcuate simple moving average sma for 42 days (two month)
sp500['42d'] = np.round(pd.rolling_mean(sp500['Close'], window=42), 2)

# Calcuate simple moving average sma for 252 days (one year)
sp500['525d'] = np.round(pd.rolling_mean(sp500['Close'], window=525), 2)

# Create the difference btw. the trende
sp500['42-252'] = sp500['42d'] - sp500['252d']

# create trading signals
SD = 50 # Threshhold

sp500['Regime'] = np.where(sp500['42-252'] > SD, 1, 0)
sp500['Regime'] = np.where(sp500['42-252'] < -SD, -1, sp500['Regime'])

#test output
sp500['Regime'].value_counts()

# calculate log returns
sp500['Market'] = np.log(sp500['Close'] / sp500['Close'].shift(1))

#returns of the trend-based trading strategy
sp500['Strategy'] = sp500['Regime'].shift(1) * sp500['Market']

# plot cumulative, continous returns
sp500[['Market', 'Strategy']].cumsum().apply(np.exp).plot(grid=True, figsize=(8, 5))




















