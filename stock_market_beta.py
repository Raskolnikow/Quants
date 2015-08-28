import numpy as np
import pandas.io.data as web
import statsmodels.api as sm


def get_stock_data(stock, start_p, end_p):
    stock = web.DataReader(stock, data_source='yahoo', start=start_p, end=end_p)
    stock['log_ret'] = np.log(stock['Adj Close'] / stock['Adj Close'].shift(1))
    stock['cum_ret'] = np.log(stock['Adj Close'] / stock['Adj Close'][0])
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
