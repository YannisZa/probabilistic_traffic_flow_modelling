import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import itertools
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import fmin

from tqdm import tqdm
from distutils.util import strtobool
from scipy.special import loggamma
from scipy.special import comb

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


""" Simulation parameters """
# Fix random seed for simulating data
simulation_seed = 2021 #None, 2001
# Define true parameters
true_parameters = [0.6,0.1,0.01]#0.005
log_true_parameters = np.log(true_parameters)
# Define parameter names
parameter_names = [r"$\alpha$",r"$\beta$",r"$\sigma^2$"]
# Get observation noise
sigma2 = true_parameters[2]
# Rho (covariate) values
rho_min = 0.1
rho_max = 40
# Number of data points
n = 150
# Number of learning parameters
num_learning_parameters = 2
# Define name of transformation
transformation = 'log'

""" MCMC parameters """
# Fix random seed for running MCMC
mcmc_seed = 2021
# Number of MCMC iterations
N = 100000
# Diagonal covariance matrix used in proposal
proposal_stds = [0.01, 0.005]
proposal_cov = 0
# K = np.array([[proposal_stds[0]**2,proposal_cov],[proposal_cov,proposal_stds[1]**2]])
K = np.diag([ps**2 for ps in proposal_stds])# [0.01125,0.00525]
# Prior sds
prior_sds = [0.01,0.005]
# Initial parameters for MCMC
p0 = [0.4,0.2]
log_p0 = np.log(p0)
# Burnin for MCMC
burnin = 2000#int(N//10)

""" Miscellaneous parameters """
# Number of standard deviations to plot parameter posterior for
num_stds = 2
# Histogram number of bins
bins = 50
# Lag number of ACF plot
lags = 200
# Flag for loading data
load_data = False


""" Simulate data without noise """
# Fix random seed
np.random.seed(simulation_seed)

# Define rho
# rho = np.concatenate([np.linspace(rho_min,rho_max,n1),np.linspace(rho_min,rho_max,n2)])
rho = np.linspace(rho_min,rho_max,n)

# Define function for simulating data
def log_simulate(p):
    return np.log(p[0])+np.log(rho)-p[1]*rho
def log_simulate_log_params(p):
    return p[0]+np.log(rho)-np.exp(p[1])*rho

# Simulate log q
log_q_true = log_simulate_log_params(log_true_parameters)

""" Sample q from Log Normal distribution """
if not load_data:
    # exp_mean = q_true / np.sqrt(1 + sigma2)
    # mean = np.log(exp_mean)
    # stdev = np.sqrt(np.log(1+sigma2))
    # q = np.array([ss.lognorm.rvs(s = stdev, loc = 0, scale = exp_mean[i]) for i in range(len(q_true))])
    # np.savetxt(f"./data/output/debugging/q.txt",q)
    # sim_mean = log_q_true - 0.5*np.log(1+sigma2)
    # sim_cov = np.eye(n)*np.log(1+sigma2)
    log_error = np.random.normal(loc=0,scale=np.sqrt(sigma2),size=n)
    print('log_error mean',np.mean(log_error))
    print('log_error std',np.var(log_error))
    #ss.multivariate_normal.rvs(mean=np.zeros(n),cov=sigma2*np.eye(n))
    log_q = log_q_true + log_error
    # log_q = ss.multivariate_normal.rvs(mean = log_q_true, cov = np.eye(n)*sigma2)
    np.savetxt(f"./data/output/debugging/log_q.txt",log_q)
else:
    # q = np.loadtxt(f"./data/output/debugging/q.txt")
    log_q = np.loadtxt(f"./data/output/debugging/log_q.txt")


# Fix random seed
np.random.seed(mcmc_seed)

def log_prior(p):
    p_trans = p #np.exp(p)
    a = 6.
    scale = 0.1
    b = 1./scale
    alpha_prior_logpdf = -(1/2)*np.log(2*np.pi*prior_sds[0]**2) - (1/2)*((np.exp(p[0])-true_parameters[0])/prior_sds[0])**2 + p[0]
    # return alpha_prior_logpdf
    a = 2.
    scale = 0.05
    b = 1./scale
    beta_prior_logpdf = -(1/2)*np.log(2*np.pi*prior_sds[1]**2) - (1/2)*((np.exp(p[1])-true_parameters[1])/prior_sds[1])**2 + p[1]
    return alpha_prior_logpdf + beta_prior_logpdf
    #return ss.geninvgauss.logpdf(p[0],p=-1,b=0.61)+ ss.geninvgauss.logpdf(p[1],p=-2,b=0.209)

def log_likelihood(p):
    # Transform phi back to alpha
    # p_trans = np.exp(p)
    # q_sim = simulate(p_trans)
    # exp_mean = q_sim / np.sqrt(1 + sigma2)
    # mean = np.log(exp_mean)
    # var = np.log(1 + sigma2)
    # stdev = np.sqrt(var)
    # return((ss.multivariate_normal.logpdf(np.log(q),mean=mean,cov=np.eye(n)*var) - np.sum(np.log(q))))
    log_q_sim = log_simulate_log_params(p)
    log_mean = log_q_sim - 0.5*np.log(1+sigma2)
    log_cov = np.eye(n)*np.log(1+sigma2)
    # return ss.multivariate_normal.logpdf(log_q,mean=log_q_sim,cov=log_cov)# - np.sum(log_q)
    return -(n/2)*np.log(2*np.pi*sigma2) -(1/(2*sigma2)) * (log_q-log_q_sim).T @ (log_q-log_q_sim)

def log_kernel(pnew,pprev):
    return 0

def log_posterior(p):
    return log_prior(p) + log_likelihood(p)

def neg_log_likelihood(p):
    return (-1)*log_likelihood(p)

def log_acceptance_ratio(pnew,pprev):
    log_pnew = log_posterior(pnew) + log_kernel(pnew,pprev)
    log_pold = log_posterior(pprev) + log_kernel(pprev,pnew)
    log_acc = log_pnew - log_pold
    # If exp floating point exceeded
    if log_acc >= 709: return 0, log_pnew, log_pold
    else: return log_acc, log_pnew, log_pold

def poisson_mean_estimator(log_samples,delta):
    J = np.random.poisson(delta)
    # Make sure J is even (to avoid negative values)
    if J % 2 != 0: J += J % 2
    print('J =',J)
    assert J <= log_samples.shape[0]
    return np.exp(delta - J * np.log(delta))*np.prod(log_samples[np.random.choice(range(log_samples.shape[0]),size=J)],axis=0)


""" MLE computation"""
log_mle = fmin(neg_log_likelihood,log_true_parameters[0:num_learning_parameters],disp=False)[0:num_learning_parameters]
print('True parameters')
for i,pname in enumerate(parameter_names):
    print(f'{pname} = {true_parameters[i]}')
print('True parameter log likelihood',log_likelihood(log_true_parameters))
print('True parameter log posterior',log_posterior(log_true_parameters))
print('Maximum likelihood estimate',np.exp(log_mle))
print('MLE likelihood',log_likelihood(log_mle))
print('MLE posterior',log_posterior(log_mle))

""" Data visualisation"""
plt.figure(figsize=(10,10))
# plt.scatter(rho,log_q,color='blue')
# plt.plot(rho,log_q_true,color='black',label='True')
# plt.plot(rho,log_simulate_log_params(log_mle),color='red',label='MLE-fitted')
plt.scatter(rho,np.exp(log_q),color='blue')
plt.plot(rho,np.exp(log_q_true),color='black',label='True')
plt.plot(rho,np.exp(log_simulate_log_params(log_mle)),color='red',label='MLE-fitted')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$q$')
plt.legend()
plt.show()


# def alpha_log_prior(p):
#     a = 6.
#     scale = 0.1
#     b = 1./scale
#     alpha_prior_logpdf = -(1/2)*np.log(2*np.pi*(prior_sds[0]**2)) - (1/2)*((np.exp(p)-true_parameters[0])/prior_sds[0])**2 + p
#     return alpha_prior_logpdf
# plt.figure(figsize=(10,10))
# print(true_parameters[0])
# print(prior_sds[0])
# # plt.scatter(rho,log_q,color='blue')
# # plt.plot(rho,log_q_true,color='black',label='True')
# # plt.plot(rho,log_simulate_log_params(log_mle),color='red',label='MLE-fitted')
# xrange = np.log(np.linspace(0.01,3,1000))
# logprior = np.array([alpha_log_prior(par) for par in xrange])
# plt.plot(xrange,logprior,color='blue')
# plt.vlines(np.log(true_parameters[0]),-1000,1)
# xmin = xrange[np.min(np.where(logprior>=-1000))]
# xmax = xrange[np.max(np.where(logprior>=-1000))]
# print(xmin)
# print(xmax)
# plt.xlim(left=xmin,right=xmax)
# plt.ylim(bottom=-1000,top=np.max(logprior))
# plt.show()

# sys.exit(1)


max_log_target = -1e9
max_log_likelihood = -1e9
max_log_likelihood_params = [-1e9,-1e9]
if not load_data:
    # Initialise MCMC-related variables
    theta = []
    theta_proposed = []
    acc = 0

    # Store necessary parameters
    p_prev = log_p0[0:num_learning_parameters]

    print('p0',np.exp(p_prev))
    print('proposal stds',np.sqrt(np.diagonal(K))[0:num_learning_parameters])
    print(f'Running MCMC with {N} iterations')

    # Loop through MCMC iterations
    for i in tqdm(range(N)):

        # Propose new sample
        p_new = p_prev + ss.multivariate_normal.rvs(np.zeros(num_learning_parameters),K[0:num_learning_parameters,0:num_learning_parameters])

        # Calculate acceptance probability
        log_acc_ratio,lt_new,lt_prev = log_acceptance_ratio(p_new,p_prev)
        acc_ratio = min(1,np.exp(log_acc_ratio))

        # Evaluate log function for proposed sample
        ll_new = log_likelihood(p_new)

        # Update maximum log-target
        if ll_new >= max_log_likelihood:
            max_log_target = lt_new
            max_log_likelihood = ll_new
            max_log_likelihood_params = p_new

        # Sample from Uniform(0,1)
        u = np.random.random()

        # Printing proposals every 0.1*Nth iteration
        if (i in [int(j/10*N) for j in range(1,11)]):
            print('p_prev',np.exp(p_prev),'lf_prev',lt_prev)
            print('p_new',np.exp(p_new),'lf_new',lt_new)

        # Accept/Reject
        # Compare log_alpha and log_u to accept/reject sample
        if acc_ratio >= u:
            if (i in [int(j/10*N) for j in range(1,11)]):
                print('p_new =',np.exp(p_new))
                print('Accepted!')
                print(f'Acceptance rate {int(100*acc / N)}%')
            # Increment accepted sample count
            acc += 1
            # Append to accepted and proposed sample arrays
            theta.append(p_new)
            theta_proposed.append(p_new)
            # Update last accepted sample
            p_prev = p_new
        else:
            if (i in [int(j/10*N) for j in range(1,11)]):
                print('Rejected...')
                print(f'Acceptance rate {int(100*acc / N)}%')
            # Append to accepted and proposed sample arrays
            theta.append(p_prev)
            theta_proposed.append(p_new)

        if i == (burnin+1):
            print('Empirical std during burnin',np.std(theta[0:burnin],axis=0))

    # Update class attributes
    theta = np.array(theta)
    theta_proposed = np.array(theta_proposed)
    print(f'Acceptance rate {int(100*acc / N)}%')

    ensure_dir(f"./data/output/debugging/")
    np.savetxt(f"./data/output/debugging/theta.txt",theta)
    np.savetxt(f"./data/output/debugging/theta_proposed.txt",theta_proposed)

else:
    theta = np.loadtxt(f"./data/output/debugging/theta.txt")
    theta_proposed = np.loadtxt(f"./data/output/debugging/theta_proposed.txt")

print('True log target',log_posterior(log_true_parameters))
print('True log likelihood',log_likelihood(log_true_parameters))
print('Max log likelihood',max_log_likelihood)
print('Argmax log likelihood',np.exp(max_log_likelihood_params))
print('Empirical std during burnin',np.std(theta[0:burnin],axis=0))

# Reshape theta
theta = theta.reshape((N,num_learning_parameters))
theta_proposed = theta_proposed.reshape((N,num_learning_parameters))

""" Parameter posterior plots"""

print(theta[-1000:-1,:])

# Compute Poisson estimator for mean of log samples
# mu_hat = poisson_mean_estimator(theta[burnin:,:],delta)
# print('Poisson estimator',mu_hat)

# Exponential log samples
# theta = np.exp(theta)

# Loop through parameter indices
for par in range(num_learning_parameters):
    # Generate figure
    fig = plt.figure(figsize=(10,8))

    # Compute posterior mean
    sample_mean = np.mean(theta[burnin:,:],axis=0)

    posterior_mean = sample_mean
    posterior_std = np.std(theta[burnin:,:],axis=0)
    # print('posterior_mean',posterior_mean)
    # print('sample_mean',sample_mean)
    print(parameter_names[par])
    print('posterior mean',posterior_mean[par])
    print('exp posterior mean',np.exp(posterior_mean[par]))
    print('posterior std',posterior_std[par])

    # Compute posterior std
    posterior_std = np.std(theta[burnin:,:],axis=0)

    # Plot parameter posterior
    freq,_,_ = plt.hist(theta[burnin:,par],bins=bins,zorder=1)

    # Add labels
    plt.title(f'Parameter posterior for {transformation} {parameter_names[par]} with burnin = {burnin}')
    plt.vlines(posterior_mean[par],0,np.max(freq),color='red',label=r'$\mu$',linewidth=2, zorder=3)
    plt.vlines(posterior_mean[par]-num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(posterior_mean[par]+num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(log_true_parameters[par],0,np.max(freq),label=f'Simulation {parameter_names[par]}',color='black',linewidth=2, zorder=2)
    # Add labels
    plt.xlabel(f'{transformation} {parameter_names[par]}')
    plt.ylabel('Sample frequency')
    # Add legend
    plt.legend()
    plt.show()


""" MCMC mixing plots"""

# Loop through parameter indices
for par in range(theta.shape[1]):

    # Generate figure
    fig = plt.figure(figsize=(10,8))

    # Add samples plot
    plt.plot(range(burnin,theta.shape[0]),theta[burnin:,par],color='blue',label='Samples',zorder=1)

    # Plot true parameters
    plt.hlines(log_true_parameters[par],xmin=burnin,xmax=(theta.shape[0]),color='black',label=f'Simulation {parameter_names[par]}',zorder=2)

    # Plot inferred mean
    plt.hlines(np.mean(theta[burnin:,par],axis=0),xmin=burnin,xmax=(theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=3)

    # Add labels
    plt.xlabel('MCMC Iterations')
    plt.ylabel(f'MCMC Samples')
    plt.title(f'Mixing for {transformation} {parameter_names[par]} with burnin = {burnin}')

    # Add legend
    plt.legend()

    plt.show()


""" MCMC ACF plots"""
# Loop through parameter indices
for par in range(theta.shape[1]):
    # Generate figure
    fig,ax = plt.subplots(1,figsize=(10,8))

    # Add ACF plot
    sm.graphics.tsa.plot_acf(theta[burnin:,par], ax=ax, lags=lags, title=f'ACF plot for {parameter_names[par]} with burnin = {burnin}')

    # Add labels
    ax.set_ylabel(f'Autocorrelation')
    ax.set_xlabel('Lags')

    plt.show()

""" MCMC space exploration plots"""

# Get number of plots
num_plots = int(comb(num_learning_parameters,2))

# Get plot combinations
parameter_indices = list(itertools.combinations(range(0,num_learning_parameters), 2))
all_parameter_names = list(itertools.combinations(parameter_names, 2))

# Avoid plotting more than 3 plots
if num_plots > 3:
    raise ValueError(f'Too many ({num_plots}) log posterior plots to handle!')
elif num_plots <= 0:
    raise ValueError(f'You cannot plot {num_plots} plots!')

# Loop through each plot
figs = []
for i in range(num_plots):
    # Get parameter indices
    index = parameter_indices[i]

    # Create figure
    fig = plt.figure(figsize=(10,8))

    # Get parameters to plot
    theta_subset = theta[burnin:,list(index)]
    theta_proposed_subset = theta_proposed[burnin:,list(index)]

    # Add samples plot
    plt.scatter(theta_subset[:,index[0]],theta_subset[:,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=5)
    plt.scatter(theta_proposed_subset[:,index[0]],theta_proposed_subset[:,index[1]],color='purple',label='Proposed',marker='x',s=50,zorder=4)

    # Plot true parameters if they exist
    plt.scatter(log_true_parameters[index[0]],log_true_parameters[index[1]],label=f'Simulation {parameter_names[index[0]]},{parameter_names[index[1]]}',marker='x',s=100,color='black',zorder=7)
    plt.scatter(log_mle[index[0]],log_mle[index[1]],label='MLE',marker='.',s=100,color='green',zorder=6)
    plt.scatter(np.mean(theta[burnin:,],axis=0)[index[0]],np.mean(theta[burnin:,],axis=0)[index[1]],label='Mean',marker='+',s=100,color='green',zorder=5)

    # Add labels
    plt.xlabel(f'{transformation} {all_parameter_names[i][index[0]]}')
    plt.ylabel(f'{transformation} {all_parameter_names[i][index[1]]}')
    # Add title
    plt.title(f'{transformation} {all_parameter_names[i][index[0]]}, {transformation} {all_parameter_names[i][index[1]]} space exploration with burnin = {burnin}')
    # Add legend
    plt.legend()

    plt.show()
