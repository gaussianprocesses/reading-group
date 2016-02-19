'''The following script is my attempt at translating r code from @mymakar into python'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import random

# Calculates the covariance matrix sigma
# @mymakar: Do something about these nasty nested loops
def calcSigma(x1, x2,l):
	''' Creating a Covariance Matrix
		INPUTS:
			X1, X2: arrays containing the x values from two separate samples
			l     : length scale parameter
		OUTPUTS:
			Sigma: a covariance matrix between x1 and X2
		--------------------------------------------------------
		Notes:
	'''
	Sigma = np.zeros((len(x1),len(x2)))
	for ii in range(len(x1)):
		for jj in range(len(x2)):
			Sigma[ii, jj] = np.exp(-0.5*(np.abs(x1[ii]-x2[jj])/float(l))**2)

	return Sigma

def sample_GP(num_functions,n_pts,mean,cov):
	''' From updated mean and covariance, sample the num_functions from the prescribed GP
		INPUTS:
			num_functions: the number of functions you want to sample from the GP
			n_pts        : the number of points to sample for each function
			mean         : the updated mean of the GP
			cov          : the updated covariance of the GP
		OUTPUTS:
			sampled_func : ndarray where the columns are the sampled functions
		------------------------------------------------------------------------
		Notes:
			Written by TWK to remove multiple instances of the same/similar loop to do this
	'''

	sampled_func = np.zeros((n_pts,num_functions))
	for ii in xrange(num_functions):
		# Sample the function from a multivariate normal, prescribed by mean and cov
		sampled_func[:,ii] = multivariate_normal.rvs(mean=mean,cov=cov)

	return sampled_func


if __name__ == '__main__':
	########################
	## Noise-free Example ##
	########################

	length_scale = 1
	x_domain = (-5,5)
	n_pts = 50
	n_samples = 3
	seed = 12345

	random.seed(seed)

	# 1. Plot some sample functions from the GP

	# Define the points at which we want to define the functions
	x_star = np.linspace(x_domain[0],x_domain[1],num=n_pts)

	# Calculate the covariance matrix between x_star and itself
	sigma = calcSigma(x_star,x_star,length_scale)

	# Generate a number of functions from the process
	sampled_values = sample_GP(n_samples,n_pts,np.zeros(n_pts),sigma)

	# Plot the result
	plt.figure()
	plt.plot(x_star,sampled_values[:,0],x_star,sampled_values[:,1],x_star,sampled_values[:,2])
	plt.fill_between(x_star,-2.5*np.ones(n_pts),2.5*np.ones(n_pts),color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.show()

	# 2. Now let's assume that we have some know data points;
	x = np.array([-4, -3, -1, 0, 2])
	y = np.array([-2, 0, 1, 2, -1])

	# Calculate the covariance matrices
	# to "condition" based on the observed (top of page 16)
	# this is the part that confused @mymakar, the following follows the text verbatim
	k_xx = calcSigma(x,x,1)
	k_xxs = calcSigma(x,x_star,1)
	k_xsx = calcSigma(x_star,x,1)
	k_xsxs = calcSigma(x_star,x_star,1)

	# Generate points according to equation 2.19
	# Update mean and covariance
	f_star_mean = k_xsx.dot(np.linalg.inv(k_xx).dot(y))
	f_star_cov = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx).dot(k_xxs))


	f_star_sampled_values = sample_GP(n_samples*10,n_pts,f_star_mean,f_star_cov)

	# Get mean and std of generated functions.
	func_mean = np.mean(f_star_sampled_values,axis=1)
	func_std = np.std(f_star_sampled_values,axis=1)
	func_lower = func_mean-(3*func_std)
	func_upper = func_mean+(3*func_std)

	# Plot the results
	plt.figure()
	plt.plot(x_star,f_star_sampled_values[:,0],x_star,f_star_sampled_values[:,1],x_star,f_star_sampled_values[:,2])
	plt.fill_between(x_star,func_lower,func_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.show()




