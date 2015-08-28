import numpy as np
import pandas.io.data as web
import statsmodels.api as sm
import matplotlib

matplotlib.use('WebAgg')

import matplotlib.pyplot as plt


def get_stock_data(stock, start_p, end_p):
    stock = web.DataReader(stock, data_source='yahoo', start=start_p, end=end_p)
    stock['log_ret'] = np.log(stock['Close'] / stock['Close'].shift(1))
    stock['cum_ret'] = np.log(stock['Close'] / stock['Close'][0])
    stock['log_ret'][0] = 0

    return stock


def lin_reg_stock(market, stock, start, end):
    # load stock data
    mkt = get_stock_data(market, start, end)
    stk = get_stock_data(stock, start, end)

    x = mkt['log_ret']
    x = sm.add_constant(x)
    y = stk['log_ret']

    mdl = sm.OLS(y, x).fit()

    return mdl

''' -------------------------------------------------------- '''

S_DATE = '1/1/2015'
E_DATA = '8/27/2015'
MARKET = 'SPY'
STOCKS = ['TSLA', 'AAPL']
stock = []

market = get_stock_data(MARKET, S_DATE, E_DATA)
for stk in STOCKS:
    stock.append(get_stock_data(stk, S_DATE, E_DATA))

plt.figure(1)
plt.grid()
plt.plot(market['cum_ret'], label=MARKET)

i=0
for stk in stock:
    plt.plot(stk['cum_ret'], label=STOCKS[i])
    i += 1

plt.legend()
plt.show()

''' -------------------------------------------------------- '''

'''mdl = lin_reg_stock(MARKET, STOCKS[0], S_DATE, E_DATA)

print mdl.summary()
plt.figure(2)
plt.grid()
plt.plot(market['log_ret'], stock['log_ret'], 'ro', alpha=0.4)
plt.show()'''
