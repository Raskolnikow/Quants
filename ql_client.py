import numpy as np
import matplotlib
import quantlab as ql

matplotlib.use('WebAgg')

import matplotlib.pyplot as plt

''' -------------------------------------------------------- '''

S_DATE = '1/1/2015'
E_DATA = '8/27/2015'
MARKET = 'SPY'
STOCKS = ['TSLA', 'AAPL', 'NFLX']
stock = []
model = []

market = ql.get_stock_data(MARKET, S_DATE, E_DATA)
for stk in STOCKS:
    stock.append(ql.get_stock_data(stk, S_DATE, E_DATA))
    model.append(ql.lin_reg_stock(MARKET, stk, S_DATE, E_DATA))

#print mdl.summary()

''' -------------------------------------------------------- '''

plt.figure(1)
plt.grid()
plt.plot(market['cum_ret'], label=MARKET)

i = 0
for stk in stock:
    plt.plot(stk['cum_ret'], label=STOCKS[i])
    i += 1

plt.legend()

i = 2
for stk in stock:
    plt.figure(i)
    plt.plot(market['log_ret'], stk['log_ret'], 'ro', alpha=0.4)
    plt.xlabel('SPY')
    plt.ylabel(STOCKS[i-2])
    i += 1

i = 2
for mdl in model:
    a = mdl.params[0]
    b = mdl.params[1]
    x = np.linspace(-0.03, 0.03, 50)
    y = a + b*x

    plt.figure(i)
    plt.grid()
    plt.plot(x, y, 'b-')
    a_txt = 'alpha = ' + "%0.2f" % a
    b_txt = 'beta = ' + "%0.2f" % b
    txt = a_txt + '\n' + b_txt
    plt.figtext(0.15, 0.8, txt)
    i += 1

plt.show()
