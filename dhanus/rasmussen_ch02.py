import numpy as np 
import scipy.stats as stats
import matplotlib as plt

def calculate_covariance(x1, x2, length_scale):
    """
    Input: x1, x2 (numpy vectors), length_scale (float, representing length_scale
    Output: S, covariance matrix, representing the covariance of x1,x2
    """
    nrows = len(x1)
    ncols = len(x2)
    S = np.zeros([nrows, ncols])
    for i in range(nrows):
        for j in range(ncols):
            z = -.5*(np.abs(x1[i]-x2[j])/length_scale)**2
            S[i,j] = np.exp(z)
    return S

# 1. plot sample functions from gaussian process 
x_star = np.linspace(-5,5,50) # define fucntion domain
sigma = calculate_covariance(x_star, x_star, 1) # calculate covariance

# generate a number of samples from the process 
nsamples = 3 
values = np.zeros([len(x_star), nsamples])
for i in range(nsamples): 
    mu = np.zeros([1,len(x_star)])
    print mu.shape
    print sigma.shape
    values[:,i] = stats.multivariate_normal(mu, sigma, 1)

# R cbind just turns it into a matrix?  https://stat.ethz.ch/R-manual/R-devel/library/base/html/cbind.html
# plot the results
plt.plot(x_star, values) 
plt.show() 

# 2. Assume known data points 
known_points = np.array([[-4,-3,-1,0,2],
                         [-2,0,1,2,-1]]) 

# define x -- what is this "x <- f$x" ?? 
sigma_xx = calculate_covariance(x,x, 1)
sigma_xxs = calculate_covariance(x,x_star, 1)
sigma_xsx = calculate_covariance(x_star,x, 1)
sigma_xsxs = calculate_covariance(x_star,x_star, 1)

y = known_points[:,1]

known_star_bar = np.dot(sigma_xsx,np.dot(sigma_xx,y)) 
known_star_sigma  = sigma_xsxs - np.dot(sigma_xsx, np.dot(sigma_xx,sigma_xxs))

#left panel of the figure 
nsamples = 3
values = np.zeros([len(x_star), nsamples])
for i in range(nsamples): 
    values[:,i] = stats.multivariate_normal(known_star_bar, known_star_sigma, 1)
    
# Plot the results including the mean function
# and constraining data points
plt.plot(x, values)
plt.show() 

# 3. Add in noise 
# The standard deviation of the noise
sigma_n = 0.1
# Recalculate the mean and covariance functions
known_bar_star_n = np.dot(sigma_xsx,(k.xx + sigma_n**2 * np.dot(np.identity(len(x)), y)))
known_star_n_sigma =sigma_xsxs - np.dot(sigma_xsx, np.dot(sigma_xx + sigman_n**2*np.identity(len(x)),sigma_xxs))
# Redraw the sample functions
values = np.zeros([len(x_star), nsamples])
for i in range(nsamples):
    values[:,i] = stats.multivariate_normal(known_star_bar_n, known_star_n_sigma, 1)
