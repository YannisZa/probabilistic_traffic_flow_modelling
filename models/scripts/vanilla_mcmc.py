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

from tqdm import tqdm
from distutils.util import strtobool
from scipy.special import loggamma
from scipy.special import comb

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


""" Simulation parameters """
# Fix random seed for simulating data
simulation_seed = 2001
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
n = 40
# Number of learning parameters
num_learning_parameters = 1

""" MCMC parameters """
# Space exploration
_alphas = np.linspace(0.5,0.7,1000)
# _betas = np.linspace(0.07,0.13,1000)
logalphas = []
alphas = []
for i in range(1000):
    logalphas.append([np.log(_alphas[i])])
    alphas.append([_alphas[i]])
# Fix random seed for running MCMC
mcmc_seed = None
# Number of MCMC iterations
N = 20000
# Step size in Random Walk proposal
beta_step = 0.3#0.01
# Diagonal covariance matrix used in proposal
K = np.diag([0.01125,0.00525])
# Initial parameters for MCMC
p0 = [0.4,0.2]
log_p0 = np.log(p0)
# Burnin for MCMC
burnin = 10000
# Delta parameter for poisson estimator
delta = 100

""" Miscellaneous parameters """
# Number of standard deviations to plot parameter posterior for
num_stds = 2
# Histogram number of bins
bins = 50
# Lag number of ACF plot
lags = 100
# Flag for loading data
load_data = False


""" Simulate data without noise """
# Fix random seed
np.random.seed(simulation_seed)

# Define rho
# rho = np.concatenate([np.linspace(rho_min,rho_max,n1),np.linspace(rho_min,rho_max,n2)])
rho = np.linspace(rho_min,rho_max,n)

# Define function for simulating data
def simulate(p):
    return p[0]*rho*np.exp(-p[1]*rho)
def log_simulate_log_params(p):
    return p[0]+np.log(rho)#-np.exp(p[1])*rho
# def simulate_log_params(p):
#     return p[0]*rho*np.exp(-np.exp(p[1])*rho)

# Simulate q
# q_true = simulate(true_parameters)
log_q_true = log_simulate_log_params(log_true_parameters)

""" Sample q from Log Normal distribution """
if not load_data:
    # exp_mean = q_true / np.sqrt(1 + sigma2)
    # mean = np.log(exp_mean)
    # stdev = np.sqrt(np.log(1+sigma2))
    # q = np.array([ss.lognorm.rvs(s = stdev, loc = 0, scale = exp_mean[i]) for i in range(len(q_true))])
    # np.savetxt(f"./data/output/debugging/q.txt",q)
    log_mean = log_q_true - 0.5*np.log(1+sigma2)
    log_cov = np.eye(n)*np.log(1+sigma2)
    log_q = ss.multivariate_normal.rvs(mean = log_mean, cov = log_cov)
    np.savetxt(f"./data/output/debugging/log_q.txt",log_q)
else:
    # q = np.loadtxt(f"./data/output/debugging/q.txt")
    log_q = np.loadtxt(f"./data/output/debugging/log_q.txt")


# Fix random seed
# np.random.seed(mcmc_seed)

def log_prior(p):
    p_trans = p #np.exp(p)
    a = 6.
    scale = 0.1
    b = 1./scale
    alpha_prior_logpdf = a*np.log(b) - loggamma(a) + a*p_trans[0] - b*np.exp(p_trans[0])
    return alpha_prior_logpdf
    # a = 2.
    # scale = 0.05
    # b = 1./scale
    # beta_prior_logpdf = a*np.log(b) - loggamma(a) + a*p_trans[1] - b*np.exp(p_trans[1])
    # return alpha_prior_logpdf + beta_prior_logpdf
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
    return ss.multivariate_normal.logpdf(log_q,mean=log_q_sim,cov=log_cov) #- np.sum(log_q)))

def log_kernel(pnew,pprev):
    return 0

def log_posterior(p):
    return log_prior(p) + log_likelihood(p)

def log_acceptance_ratio(pnew,pprev):
    log_acc = log_posterior(pnew) + log_kernel(pnew,pprev) - log_posterior(pprev) - log_kernel(pprev,pnew)
    # If exp floating point exceeded
    if log_acc > 709: return 0
    else: return log_acc

# def log_acceptance_ratio(pnew,pprev):
#     ltnew = log_prior(pnew) + log_likelihood(pnew)
#     ltprev = log_prior(pprev) + log_likelihood(pprev)
#     return min(ltnew-ltprev,0)

def poisson_mean_estimator(log_samples,delta):
    J = np.random.poisson(delta)
    # Make sure J is even (to avoid negative values)
    if J % 2 != 0: J += J % 2
    print('J =',J)
    assert J <= log_samples.shape[0]
    return np.exp(delta - J * np.log(delta))*np.prod(log_samples[np.random.choice(range(log_samples.shape[0]),size=J)],axis=0)

plt.figure(figsize=(10,10))
# plt.scatter(rho,q,color='blue')
# plt.scatter(rho,log_q,color='blue')
# plt.plot(rho,log_q_true,color='red')
plt.scatter(rho,np.exp(log_q),color='blue')
plt.plot(rho,np.exp(log_q_true),color='red')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$\log q$')
plt.show()

# print('True parameters')
# for i,pname in enumerate(parameter_names):
#     print(f'{pname} = {true_parameters[i]}')
# print('True log_prior',log_prior(log_true_parameters))
# print('True log_likelihood',log_likelihood(log_true_parameters))
# print('True log target',log_posterior(log_true_parameters))

""" Space exploration"""
# log_likelihood_vec = np.vectorize(log_likelihood, otypes=[list])
# log_posterior_vec = np.vectorize(log_posterior, otypes=[list])
# print(log_likelihood_vec([[0.1],[0.1],[0.1]]))
# print(logalphas)
loglike = [log_likelihood(a) for a in logalphas]
logpost = [log_posterior(a) for a in logalphas]
print('empirical MLE',alphas[np.argmax(loglike)])
print('empirical MLE log likelihood',log_likelihood(logalphas[np.argmax(loglike)]))
print('empirical MAP',alphas[np.argmax(logpost)])
print('empirical MAP log posterior',log_posterior(logalphas[np.argmax(logpost)]))
print('true parameter',true_parameters[0])
print('true parameter log likelihood',log_likelihood(log_true_parameters))
print('true parameter log posterior',log_posterior(log_true_parameters))
plt.figure(figsize=(10,10))
plt.title('Log distributions in log-parameter space')
# /np.max(np.exp(loglike)
# plt.plot(np.linspace(true_parameters[0]-0.1,true_parameters[0]+0.1,1000),np.exp(logpost)/np.max(np.exp(logpost)),color='blue',label='Log posterior')
plt.plot(logalphas,loglike,color='red',label='Log likelihood')
plt.vlines(log_true_parameters[0],np.min(loglike),np.max(loglike),color='black',label='True')
plt.vlines(logalphas[np.argmax(loglike)],np.min(loglike),np.max(loglike),color='red',label='MLE')
plt.legend()
plt.xlim(np.min(logalphas),np.max(logalphas))
plt.show()

# sys.exit(1)
max_log_target = -1e9
max_log_likelihood = -1e9
max_log_likelihood_params = [-1]
if not load_data:
    # Initialise MCMC-related variables
    theta = []
    theta_proposed = []
    acc = 0

    # Store necessary parameters
    p_prev = log_p0[0:num_learning_parameters]

    print('p0',np.exp(p_prev))
    print('proposal stds',np.sqrt(np.diagonal(K)*beta_step**2)[0:num_learning_parameters])
    print(f'Running MCMC with {N} iterations')

    # Loop through MCMC iterations
    for i in tqdm(range(N)):

        # Evaluate log function for current sample
        lt_prev = log_posterior(p_prev)
        ll_prev = log_likelihood(p_prev)

        # Propose new sample
        p_new = p_prev + beta_step * ss.multivariate_normal.rvs(np.zeros(num_learning_parameters),K[0:num_learning_parameters,0:num_learning_parameters])

        # Evaluate log function for proposed sample
        lt_new = log_posterior(p_new)
        ll_new = log_likelihood(p_new)

        # Update maximum log-target
        if ll_new >= max_log_likelihood:
            max_log_target = lt_new
            max_log_likelihood = ll_new
            max_log_likelihood_params = p_new

        # Printing proposals every 0.1*Nth iteration
        if (i in [int(j/10*N) for j in range(1,11)]):
            print('p_prev',np.exp(p_prev),'ll_prev',ll_prev,'lf_prev',lt_prev)
            print('p_new',np.exp(p_new),'ll_new',ll_new,'lf_new',lt_new)

        # Calculate acceptance probability
        log_acc_ratio = log_acceptance_ratio(p_new,p_prev)
        acc_ratio = min(1,np.exp(log_acc_ratio))

        # Sample from Uniform(0,1)
        u = ss.uniform.rvs(0,1)

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
            print('Empirical variances during burnin',np.var(theta[0:burnin]))

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

print('True log target',log_posterior(true_parameters))
print('True log likelihood',log_likelihood(true_parameters))
print('Max log likelihood',max_log_likelihood)
print('Argmax log likelihood',max_log_likelihood_params)
print('Empirical variances during burnin',np.var(theta[0:burnin]))

# Reshape theta
theta = theta.reshape((N,num_learning_parameters))
theta_proposed = theta_proposed.reshape((N,num_learning_parameters))

""" Parameter posterior plots"""

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
    sigma2 = true_parameters[-1]
    var = np.log(1+sigma2)

    posterior_mean = sample_mean
    # print('posterior_mean',posterior_mean)
    # print('sample_mean',sample_mean)

    # Compute posterior std
    posterior_std = np.std(theta[burnin:,:],axis=0)

    # Plot parameter posterior
    freq,_,_ = plt.hist(theta[burnin:,par],bins=bins,zorder=1)

    # Add labels
    plt.title(f'Parameter posterior for {parameter_names[par]} with burnin = {burnin}')
    plt.vlines(posterior_mean[par],0,np.max(freq),color='red',label=r'$\mu$',linewidth=2, zorder=3)
    plt.vlines(posterior_mean[par]-num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(posterior_mean[par]+num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(log_true_parameters[par],0,np.max(freq),label='True',color='black',linewidth=2, zorder=2)
    # Add labels
    plt.xlabel(f'{parameter_names[par]}')
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
    plt.hlines(log_true_parameters[par],xmin=burnin,xmax=(theta.shape[0]),color='black',label='True',zorder=2)

    print(parameter_names[par])
    print('posterior mean',np.exp(np.mean(theta[burnin:,par],axis=0)))

    # Plot inferred mean
    plt.hlines(np.mean(theta[burnin:,par],axis=0),xmin=burnin,xmax=(theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=3)

    # Add labels
    plt.xlabel('MCMC Iterations')
    plt.ylabel(f'MCMC Samples')
    plt.title(f'Mixing for {parameter_names[par]} with burnin = {burnin}')

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
    ax.set_ylabel(f'{parameter_names[par]}')
    ax.set_xlabel('Lags')

    plt.show()

""" MCMC space exploration plots"""

# Get number of plots
num_plots = int(comb(num_learning_parameters,2))

# Get plot combinations
parameter_indices = list(itertools.combinations(range(0,num_learning_parameters), 2))
parameter_names = list(itertools.combinations(parameter_names, 2))

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

    # plt.xlim([float(plot_limits['xmin']),float(plot_limits['xmax'])])
    # plt.ylim([float(plot_limits['ymin']),float(plot_limits['ymax'])])

    # Plot true parameters if they exist
    plt.scatter(log_true_parameters[index[0]],log_true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=7)

    # Add labels
    plt.xlabel(f'{parameter_names[i][index[0]]}')
    plt.ylabel(f'{parameter_names[i][index[1]]}')
    # Add title
    plt.title(f'{parameter_names[i][index[0]]},{parameter_names[i][index[1]]} space exploration with burnin = {burnin}')
    # Add legend
    plt.legend()

    plt.show()
