'''The following script is my attempt at translating r code from @mymakar into python'''

import autograd.numpy as np   # Thinly-wrapped version of Numpy
import autograd.numpy.random as npr
from autograd import grad, value_and_grad
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.optimize import minimize
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
	length_scale = l
	diffs = np.expand_dims(x1 /length_scale,1)\
		 - np.expand_dims(x2 /length_scale,0)
	return np.exp(-0.5 * np.sum(diffs**2,axis=2))

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

def calculate_func_mean_and_variance(func_samples):
	''' Simple helper function that will calulate the mean and variance 
		of a collection of functions
		INPUT:
			func_samples: an nd-array of functions, functions are along the columns
		OUTPUT:
			func_mean  : the mean function of func_samples
			func_lower : the lower bound (statistically) of func_samples
			func_upper : the upper bound (statistically) of func_samples
		----------------------------------------------------------------------------
		Notes:
			I use 3sigma variance, where 99.7 of the spread is accounted for.
	'''

	func_mean = np.mean(func_samples,axis=1)
	func_std = np.std(func_samples,axis=1)
	func_lower = func_mean-(3*func_std)
	func_upper = func_mean+(3*func_std)

	return func_mean, func_lower, func_upper



if __name__ == '__main__':
	
	########################
	## Noise-free Example ##
	########################

	length_scale = 1.0
	x_domain = (-5,5)
	n_pts = 50
	n_samples = 3
	seed = 12345
	rs = npr.RandomState(0)

	num_params = 1

	random.seed(seed)

	# 1. Plot some sample functions from the GP

	# Define the points at which we want to define the functions
	x_star = np.reshape(np.linspace(x_domain[0],x_domain[1],num=n_pts),(n_pts,1))

	# Calculate the covariance matrix between x_star and itself
	sigma = calcSigma(x_star,x_star,length_scale)

	# Generate a number of functions from the process
	sampled_values = sample_GP(n_samples,n_pts,np.zeros(n_pts),sigma)

	# Plot the result
	# import pdb
	# pdb.set_trace()
	plt.figure()
	plt.plot(x_star,sampled_values[:,0],x_star,sampled_values[:,1],x_star,sampled_values[:,2])
	plt.fill_between(x_star.flatten(),-2.5*np.ones(n_pts),2.5*np.ones(n_pts),color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title('Sampling from Prior')
	plt.show()

	# 2. Now let's assume that we have some know data points;
	x = np.reshape(np.array([-4, -3, -1, 0, 2]),(5,1))
	y = np.array([-2, 0, 1, 2, -1])

	# Calculate the covariance matrices
	# to "condition" based on the observed (top of page 16)
	# this is the part that confused @mymakar, the following follows the text verbatim
	k_xx = calcSigma(x,x,length_scale)
	k_xxs = calcSigma(x,x_star,length_scale)
	k_xsx = calcSigma(x_star,x,length_scale)
	k_xsxs = calcSigma(x_star,x_star,length_scale)

	# Generate points according to equation 2.19
	# Update mean and covariance
	f_star_mean = k_xsx.dot(np.linalg.inv(k_xx).dot(y))
	f_star_cov = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx).dot(k_xxs))


	f_star_sampled_values = sample_GP(n_samples*10,n_pts,f_star_mean,f_star_cov)

	# Get mean and std of generated functions.
	func_mean, func_lower, func_upper = calculate_func_mean_and_variance(f_star_sampled_values)

	# Plot the results
	plt.figure()
	plt.plot(x_star,f_star_sampled_values[:,0],x_star,f_star_sampled_values[:,1],x_star,f_star_sampled_values[:,2])
	plt.fill_between(x_star.flatten(),func_lower,func_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction with noise-free observations")
	plt.show()

	########################
	## With Noise Example ##
	########################

	# Standard deviation of the noise
	sigma_n = 0.1

	# Update the mean and covariance from equations 2.22-2.24
	f_bar_star_mean = k_xsx.dot(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])).dot(y))
	f_bar_star_cov = k_xsxs - k_xsx.dot(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])).dot(k_xxs))

	# Redraw the sample functions
	f_bar_star_sampled_values = sample_GP(n_samples*10,n_pts,f_bar_star_mean,f_bar_star_cov)

	# Get mean and spread of newly sampled values
	func_bar_mean, func_bar_lower, func_bar_upper = calculate_func_mean_and_variance(f_bar_star_sampled_values)

	# Plot the results
	plt.figure()
	plt.plot(x_star,f_bar_star_sampled_values[:,0],x_star,f_bar_star_sampled_values[:,1],x_star,f_bar_star_sampled_values[:,2])
	plt.fill_between(x_star.flatten(),func_bar_lower,func_bar_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction using noisy observations")
	plt.show()

	###################################
	## Marginal likelihood (p(y| X)) ##
	###################################

	def marg_likelihood(x, y, l):
		k_xx = calcSigma(x,x,l)
		marg_data = 0.5* np.dot(y.T,np.dot(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])),y)) - 0.5 * \
			np.log(np.linalg.det(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])))) - (len(y)*0.5) * np.log(2*np.pi) 

		return -1.0*marg_data


	###################################
	####         Gradient          ####
	###################################

	g_ml = lambda l: marg_likelihood(x,y,l)

	init_params = 0.1 * rs.randn(num_params)
	grad_ml = grad(g_ml)
	cov_params = minimize(value_and_grad(g_ml),init_params,jac=True,
						method = 'CG')

	print marg_likelihood(x,y,length_scale)
	print grad_ml(length_scale)
	print "Initial Parameters: ", init_params
	print "Optimized Parameters: ", cov_params.x

	opt_length_scale = np.exp(cov_params.x[0])

	# Calculate the covariance matrices with optimized length scale
	# to "condition" based on the observed (top of page 16)
	# this is the part that confused @mymakar, the following follows the text verbatim
	ok_xx = calcSigma(x,x,opt_length_scale)
	ok_xxs = calcSigma(x,x_star,opt_length_scale)
	ok_xsx = calcSigma(x_star,x,opt_length_scale)
	ok_xsxs = calcSigma(x_star,x_star,opt_length_scale)


	# Update the mean and covariance from equations 2.22-2.24
	of_bar_star_mean = ok_xsx.dot(np.linalg.inv(ok_xx+ (sigma_n**2)*np.identity(ok_xx.shape[0])).dot(y))
	of_bar_star_cov = ok_xsxs - ok_xsx.dot(np.linalg.inv(ok_xx+ (sigma_n**2)*np.identity(ok_xx.shape[0])).dot(ok_xxs))

	# Redraw the sample functions
	of_bar_star_sampled_values = sample_GP(n_samples*10,n_pts,of_bar_star_mean,of_bar_star_cov)

	# Get mean and spread of newly sampled values
	ofunc_bar_mean, ofunc_bar_lower, ofunc_bar_upper = calculate_func_mean_and_variance(of_bar_star_sampled_values)

	# Plot the results
	plt.figure()
	plt.plot(x_star,of_bar_star_sampled_values[:,0],x_star,of_bar_star_sampled_values[:,1],x_star,of_bar_star_sampled_values[:,2])
	plt.fill_between(x_star.flatten(),ofunc_bar_lower,ofunc_bar_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction using noisy observations, with optimized length scale")
	plt.show()











