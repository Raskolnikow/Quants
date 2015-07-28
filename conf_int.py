import numpy as np
from scipy.stats import norm
from scipy.stats import t

def _sample_std(sample):
	''' Calculates the standard deviation of a sample

		Parameters
		==========
		sample: array
			sample data

		Returns
		=======
		value :	float
			the sample standard deviation of the sample
	'''

	std = np.array(sample).std()
	mean = np.array(sample).mean()

	sq = map(lambda x: (x-mean)**2, sample)
	var = reduce(lambda x,y: x+y, sq)
	var = var/(len(sample)-1)
	std = np.sqrt(var)

	return std


def mu_intervall(sample, var, gamma):
	'''
		calcuates the confidence intervall for the mean of a population.

		Parameters
		==========
		sample: array
			sample data
		var: 	float
			variance of sample. 0 if not known. will be calculated by an estimator
		gamme: 	float
			confidence

		Returns
		=======
		value :		tuple (a,b)
			confidence intervall as a tuple
	'''

	s_mean = np.array(sample).mean()
	

	if(var == 0):
		std = _sample_std(sample)
		q = t.ppf((1+gamma)/2.0, len(sample)-1)
	else:
		std = np.sqrt(var)
		q = norm.ppf((1+gamma)/2.0)

	c = q * std/np.sqrt(len(sample))
	c1 = s_mean - c
	c2 = s_mean + c
	return (c1, c2)