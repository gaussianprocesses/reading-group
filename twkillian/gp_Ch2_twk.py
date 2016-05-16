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

	length_scale = 1
	x_domain = (-5,5)
	n_pts = 100
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
	# plt.figure()
	# plt.plot(x_star,sampled_values[:,0],x_star,sampled_values[:,1],x_star,sampled_values[:,2])
	# plt.fill_between(x_star,-2.5*np.ones(n_pts),2.5*np.ones(n_pts),color='0.15',alpha=0.25)
	# plt.xlabel('input, x')
	# plt.ylabel('output, f(x)')
	# plt.title('Sampling from Prior')
	# plt.show()

	# 2. Now let's assume that we have some know data points;
	x = np.array([-4, -3, -1, 0, 2])
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
	# plt.figure()
	# plt.plot(x_star,f_star_sampled_values[:,0],x_star,f_star_sampled_values[:,1],x_star,f_star_sampled_values[:,2])
	# plt.fill_between(x_star,func_lower,func_upper,color='0.15',alpha=0.25)
	# plt.xlabel('input, x')
	# plt.ylabel('output, f(x)')
	# plt.title("Prediction with noise-free observations")
	# plt.show()


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
	# plt.figure()
	# plt.plot(x_star,f_bar_star_sampled_values[:,0],x_star,f_bar_star_sampled_values[:,1],x_star,f_bar_star_sampled_values[:,2])
	# plt.fill_between(x_star,func_bar_lower,func_bar_upper,color='0.15',alpha=0.25)
	# plt.xlabel('input, x')
	# plt.ylabel('output, f(x)')
	# plt.title("Prediction using noisy observations")
	# plt.show()

	###################################
	## Marginal likelihood (p(y| X)) ##
	###################################

	marg_data = 0.5* y.T.dot(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])).dot(y)) - 0.5 * \
		np.log(np.linalg.det(np.linalg.inv(k_xx+ (sigma_n**2)*np.identity(k_xx.shape[0])))) - (len(y)*0.5) * np.log(2*np.pi) 



	#############################################
	## Monte Carlo Approach to Uncertain Input ##
	#############################################

	# Now consider the instance where the test inputs are randomly determined themselves. 
	# Here we'll loop through a number of times to see how the the predicted functions change.

	n_mc = 100

	unc_sampled_funcs = np.zeros((n_pts,n_samples*10,n_mc))

	for i_mc in xrange(n_mc):

		# Generate randomized x_star
		temp_x_star = np.random.normal(x_star,0.25)

		# Calculate all of the covariance matrices and etc.
		temp_k_xx = calcSigma(x,x,length_scale)
		temp_k_xxs = calcSigma(x,temp_x_star,length_scale)
		temp_k_xsx = calcSigma(temp_x_star,x,length_scale)
		temp_k_xsxs = calcSigma(temp_x_star,temp_x_star,length_scale)

		# Generate points according to equation 2.19
		# Update mean and covariance
		temp_f_bar_star_mean = temp_k_xsx.dot(np.linalg.inv(temp_k_xx+ (sigma_n**2)*np.identity(temp_k_xx.shape[0])).dot(y))
		temp_f_bar_star_cov = temp_k_xsxs - temp_k_xsx.dot(np.linalg.inv(temp_k_xx+ (sigma_n**2)*np.identity(temp_k_xx.shape[0])).dot(temp_k_xxs))

		# Redraw the sample functions
		unc_sampled_funcs[:,:,i_mc] = sample_GP(n_samples*10,n_pts,temp_f_bar_star_mean,temp_f_bar_star_cov)

	# import pdb
	# pdb.set_trace()

	# Take the mean over all MC determined posterior draws
	# unc_averaged_values = np.mean(unc_sampled_funcs,axis=2)
	unc_averaged_values = unc_sampled_funcs[:,:,-1]

	# Get mean and spread of newly sampled values
	unc_func_bar_mean, unc_func_bar_lower, unc_func_bar_upper = calculate_func_mean_and_variance(unc_averaged_values)

	# Plot the results
	# plt.figure()
	# plt.plot(temp_x_star,unc_averaged_values[:,0],temp_x_star,unc_averaged_values[:,10],temp_x_star,unc_averaged_values[:,20])
	# plt.fill_between(temp_x_star,unc_func_bar_lower,unc_func_bar_upper,color='0.15',alpha=0.25)
	# plt.xlabel('input, x')
	# plt.ylabel('output, f(x)')
	# plt.title("Prediction using noisy observations with uncertain input")
	# plt.show()

	# Plot all four together
	plt.figure(figsize=(18,16))
	plt.subplot(2,2,1)
	plt.plot(x_star,sampled_values[:,0],x_star,sampled_values[:,1],x_star,sampled_values[:,2])
	plt.fill_between(x_star,-2.5*np.ones(n_pts),2.5*np.ones(n_pts),color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title('Sampling from Prior')
	plt.subplot(2,2,2)
	plt.plot(x_star,f_star_sampled_values[:,0],x_star,f_star_sampled_values[:,1],x_star,f_star_sampled_values[:,2])
	plt.fill_between(x_star,func_lower,func_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction with noise-free observations")
	plt.subplot(2,2,3)
	plt.plot(x_star,f_bar_star_sampled_values[:,0],x_star,f_bar_star_sampled_values[:,1],x_star,f_bar_star_sampled_values[:,2])
	plt.fill_between(x_star,func_bar_lower,func_bar_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction using noisy observations")
	plt.subplot(2,2,4)
	plt.plot(temp_x_star,unc_averaged_values[:,0],temp_x_star,unc_averaged_values[:,1],temp_x_star,unc_averaged_values[:,2])
	plt.fill_between(temp_x_star,unc_func_bar_lower,unc_func_bar_upper,color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction using noisy observations with uncertain input")
	plt.show()





 