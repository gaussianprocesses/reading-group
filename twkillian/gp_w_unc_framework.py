'''The following script is my attempt at fleshing out a generic framework for GPs with uncertain inputs'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import gaussian_kde
import random

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


def plotGaussian(x,m,v):
	'''Takes in approximated mean and variance along with input values
	 and provides Gaussian distribution output values'''

	return (2*v*np.pi)**(-0.5) * np.exp(-(1.0/(2*v))*(x-m)**2)

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

	n_train = 10 # Number of training points
	noise_var = 0.025 # Process noise of observing the true function

	length_scale = 1.0
	x_domain = (-5,5)
	n_pts = 100
	n_samples = 3

	full_x = np.linspace(x_domain[0],x_domain[1],n_pts).reshape((-1,1))

	seed = 222

	# true_func = lambda xx: xx**2 * np.cos(xx)/xx
	# true_func = lambda xx: np.sqrt(np.abs(xx)) * np.sin(xx)**2
	true_func = lambda xx: np.exp(-np.sin(xx)/xx)

	# Generate training points
	x_train = np.random.uniform(low=-5.,high=5.,size=n_train).reshape((-1,1))
	y_train = true_func(x_train) + np.random.normal(loc=0.0,scale=noise_var,size=(n_train,1))

	##################################
	##  DETERMINISTIC PRIOR BASE GP ##
	##################################

	# Train Gaussian Process (only considering examples where noise is present)
	K = calcSigma(x_train,x_train,length_scale) # Get training covariance
	ks = calcSigma(x_train,full_x,length_scale) # Get covariance between train and test points
	kss = calcSigma(full_x,full_x,length_scale) # Get test covariance

	# import pdb
	# pdb.set_trace()

	Omg = np.linalg.inv( K + ((noise_var/2.)**2*np.identity(n_train)) )
	Beta = np.dot(Omg,y_train).reshape((-1,1))


	post_mean = np.dot(ks.T,Beta)
	post_var = kss - ks.T.dot(Omg.dot(ks))

	# Sample from prior
	prior_sampled_values = sample_GP(n_samples,n_pts,np.zeros(n_pts),kss)

	# Sample from posterior predictive distribution
	post_sampled_values = sample_GP(n_samples*100,n_pts,post_mean.flatten(),post_var)

	# Get mean and spread of newly sampled values
	f_post_mean, f_post_lower, f_post_upper = calculate_func_mean_and_variance(post_sampled_values)

	# Plot all prior and posterior together
	plt.figure(figsize=(14,6))
	plt.subplot(1,2,1)
	plt.plot(full_x,prior_sampled_values[:,0],full_x,prior_sampled_values[:,1],full_x,prior_sampled_values[:,2])
	plt.plot(full_x,true_func(full_x),'k-',lw=3,label='True Function',alpha=0.35) # Plot the underlying function generating the data
	plt.plot(x_train,y_train,'r*',markersize=10,label='Training Input',alpha=0.35) 
	plt.fill_between(full_x.flatten(),-2.5*np.ones(n_pts),2.5*np.ones(n_pts),color='0.15',alpha=0.25)
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title('Sampling from Prior')
	plt.subplot(1,2,2)
	plt.plot(full_x,post_sampled_values[:,0],full_x,post_sampled_values[:,1],full_x,post_sampled_values[:,2])
	plt.fill_between(full_x.flatten(),f_post_lower,f_post_upper,color='0.15',alpha=0.25)
	plt.plot(full_x,true_func(full_x),'k-',lw=3,label='True Function',alpha=0.35) # Plot the underlying function generating the data
	plt.plot(x_train,y_train,'r*',markersize=10,label='Training Input',alpha=0.35) 
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("Prediction with noisy observations")
	plt.show()

	#########################
	##  UNCERTAIN PRIOR GP ##
	#########################

	# Define input distribution
	unc_mean = 2.5
	unc_var = 0.5

	sampled_unc_inputs = np.random.normal(unc_mean,unc_var,size=(10*n_pts,1))

	## MONTE-CARLO METHOD - With sampled pts from input distribution, gather sampled output from GP.
	#####################
	#-------Need to rethink the sampling of the GP

	K_unc_mc = calcSigma(x_train,sampled_unc_inputs,length_scale)
	K_unc_mc2 = calcSigma(sampled_unc_inputs,sampled_unc_inputs,length_scale)

	ks = calcSigma(x_train,full_x,length_scale) # Get covariance between train and test points
	kss = calcSigma(full_x,full_x,length_scale) # Get test covariance

	post_mc_mean = np.dot(K_unc_mc.T,Beta)
	post_mc_var = K_unc_mc2 - np.dot(K_unc_mc.T,np.dot(Omg,K_unc_mc))

	post_mc_sampled_values = sample_GP(1,10*n_pts,post_mc_mean.flatten(),post_mc_var).flatten()

	post_MC_mean = post_mc_sampled_values.mean()
	post_MC_var = post_mc_sampled_values.var()

	mc_density = gaussian_kde(post_mc_sampled_values)
	mc_unc_xs = np.linspace(-4,4,10*n_pts)


	## EXACT METHOD
	###############

	# Calculate L and l
	# import pdb
	# pdb.set_trace()

	K_unc = calcSigma(np.array(unc_mean).reshape((-1,1)),x_train,length_scale).flatten() # Gather gaussian kernel of how input mean relates to training inputs
	K_unc_adj = calcSigma([unc_mean],x_train,-1.0*np.sqrt((length_scale*(length_scale+unc_var))/unc_var)) # Generate adjusted kernel-like distribution

	ll = ((1+(unc_var/length_scale))**(-0.5)*K_unc*K_unc_adj).T # Calculate l, the adjusted Kernel

	L = np.zeros((n_train,n_train))
	for ii in range(n_train):
		for jj in range(n_train):
			xd = np.mean([x_train[ii],x_train[jj]])
			L[ii,jj] = K_unc[ii]*K_unc[jj]*(1+2*(unc_var/length_scale))**(-0.5) * np.exp(0.5*(unc_var/((0.5*length_scale + unc_var)*(0.5*length_scale)))*(unc_mean-xd)**2)

	# Generate posterior mean and variance
	post_unc_mean = (Beta.T.dot(ll)).flatten()[0]

	post_unc_var = (1 - np.trace( np.dot(Omg,L)) ) + ( np.trace( np.dot( np.dot(Beta,Beta.T), L-np.dot(ll,ll.T) ) ) )

	# Plot posterior output distribution
	exact_unc_xs = np.linspace(-4,4,10*n_pts)
	plt.figure(figsize=(16,18))
	plt.subplot(2,2,1)
	plt.plot(plotGaussian(exact_unc_xs,post_unc_mean,post_unc_var),exact_unc_xs,'g-',lw=3)
	# plt.ylim([-4, 4])
	plt.title("Posterior Output Distribution--Exact Method")
	plt.ylabel("output, f(x)")
	plt.subplot(2,2,2)
	plt.plot(full_x,post_sampled_values[:,0],full_x,post_sampled_values[:,1],full_x,post_sampled_values[:,2])
	plt.fill_between(full_x.flatten(),f_post_lower,f_post_upper,color='0.15',alpha=0.25)
	plt.plot(full_x,true_func(full_x),'k-',lw=3,label='True Function',alpha=0.35) # Plot the underlying function generating the data
	plt.plot(x_train,y_train,'r*',markersize=10,label='Training Input',alpha=0.35) 
	plt.ylim([-4,4])
	plt.xlabel('input, x')
	plt.ylabel('output, f(x)')
	plt.title("GP Prediction with noisy observations")
	plt.subplot(2,2,3)
	plt.plot(mc_density(mc_unc_xs),mc_unc_xs,'g--',lw=3)
	plt.title('Posterior Output Distribution--Monte-Carlo Method')
	plt.subplot(2,2,4)
	plt.plot(full_x,plotGaussian(full_x,unc_mean,unc_var),'b-',lw=3)
	plt.title('Input Distribution')
	plt.xlabel('input, x')
	plt.show()



	# Generate MC...

