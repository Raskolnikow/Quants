import numpy as np
import numpy.io.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm

# Get the data
spy = web.DataReader('SPY', data_source='yahoo',
                     start='1/1/2015',
                     end='8/27/2015')
aapl = web.DataReader(
    'AAPL', data_source='yahoo', start='1/1/2015', end='8/27/2015')

# calc daily returns
spy['daily_ret'] = spy['Close'] / spy['Close'].shift(1)
aapl['daily_ret'] = aapl['Close'] / aapl['Close'].shift(1)

# plot daily returns SPY vs. AAPL as scatter plot
plt.plot(spy['daily_ret'], aapl['daily_ret'], 'ro')

# create linear model ( regression )
X = spy['daily_ret']
X = sm.add_constant(X)

Y = aapl['daily_ret']

# since the first value is NaN, copy the second value into it
X['daily_ret'][0] = X['daily_ret'][1]
Y[0] = Y[1]

model = sm.OLS(Y, X)
result = model.fit()

# show the results
result.summary()

alpha = result.params[0]
beta = result.params[1]

# plot the regression line
X_fit = np.linspace(0.96, 1.03, 50)
Y_fit = alpha + beta * X_fit

plt.plot(X_fit, Y_fit, 'b-')
