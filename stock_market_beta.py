import numpy as np
import pandas.io.data as web
import statsmodels.api as sm
import matplotlib

matplotlib.use('MacOSX')

import matplotlib.pyplot as plt


def lin_reg_stock(stock):
    # load stock data
    spy = web.DataReader('SPY', data_source='yahoo',
                         start='1/1/2015', end='8/27/2015')
    nflx = web.DataReader(
        stock, data_source='yahoo', start='1/1/2015', end='8/27/2015')

    spy['log_ret'] = np.log(spy['Close'] / spy['Close'].shift(1))
    nflx['log_ret'] = np.log(nflx['Adj Close'] / nflx['Adj Close'].shift(1))

    spy['cum_ret'] = np.log(spy['Close'] / spy['Close'][0])
    nflx['cum_ret'] = np.log(nflx['Adj Close'] / nflx['Adj Close'][0])

    spy['log_ret'][0] = 0
    nflx['log_ret'][0] = 0

    x = spy['log_ret']
    x = sm.add_constant(x)
    y = nflx['log_ret']

    mdl = sm.OLS(y, x).fit()
    print mdl.summary()

    plt.figure(1)
    plt.grid()
    plt.plot(spy['cum_ret'], label='SPY')
    plt.plot(nflx['cum_ret'], label=stock)
    plt.legend()

    plt.figure(2)
    plt.grid()
    plt.plot(spy['log_ret'], nflx['log_ret'], 'ro', alpha=0.4)
    plt.show()

lin_reg_stock('XOM')
