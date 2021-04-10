import os
import copy
import itertools
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.special import comb
from scipy.optimize import fmin
from scipy.special import loggamma

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


""" Simulation parameters """
# Fix random seed for simulating data
simulation_seed = 2021 # None
# Define true parameters
true_parameters = [0.6,0.1,1]
# Define parameter names
parameter_names = [r"$\alpha$",r"$\beta$",r"$\sigma^2$"]
# Get observation noise
sigma2 = true_parameters[2]
print('sigma2',sigma2)
# Rho (covariate) values
rho_min = 0
rho_max = 40
# Number of data points
n = 100
# Number of learning parameters
num_learning_parameters = 1

""" MCMC parameters """
# Fix random seed for running MCMC
mcmc_seed = 2021
# Number of MCMC iterations
N = 100000
# Diagonal covariance matrix used in proposal
proposal_std = 0.01
K = np.diag([proposal_std**2])#0.00018765662769967983
# Initial parameters for MCMC - if None sample from prior
p0 = [0.1]
# Prior sd
prior_sd = 0.05
# Burnin for MCMC
burnin = 1000 #int(N//2)

""" Miscellaneous parameters """
# Grid parameter search
alphas = np.linspace(0.55,0.65,1000)
# Number of standard deviations to plot parameter posterior for
num_stds = 2
# Histogram number of bins
bins = 50
# Lag number of ACF plot
lags = 100
# Flag for loading data
load_data = True

""" Simulate data without noise """
# Fix random seed
# np.random.seed(simulation_seed)

# Define rho
rho = np.linspace(rho_min,rho_max,n)

# Define function for simulating data
def simulate(p):
    return p[0]*rho

# Simulate q
q_true = simulate(true_parameters)

""" Sample q from a Normal distribution """
if not load_data:
    error = np.random.normal(loc=0,scale=np.sqrt(sigma2),size=n)
    print('error mean',np.mean(error))
    print('error std',np.var(error))
    #ss.multivariate_normal.rvs(mean=np.zeros(n),cov=sigma2*np.eye(n))
    q = q_true + error
    np.savetxt(f"./data/output/debugging/simplified_q.txt",q)
else:
    q = np.loadtxt(f"./data/output/debugging/simplified_q.txt")

# Fix random seed
# np.random.seed(mcmc_seed)

""" Define functions """

def log_prior(p):
    # a = 6.
    # scale = 0.1
    # b = 1./scale
    # alpha_prior_logpdf = a*np.log(b) - loggamma(a) + a*p_trans[0] - b*np.exp(p_trans[0])
    # return alpha_prior_logpdf
    # print( ss.gamma.logpdf(p[0],a=6.,scale=0.1) )
    # print( a * np.log(b) - loggamma(a) + (a-1) * np.log(p[0]) - b * p[0] )
    # return a * np.log(b) - loggamma(a) + (a-1) * np.log(p[0]) - b * p[0]
    # print(-(1/2)*np.log(2*np.pi*prior_sd**2) - (1/2)*((p[0]-true_parameters[0])/prior_sd)**2)
    # print( ss.norm.logpdf(p[0],loc=true_parameters[0],scale=prior_sd) )
    return -(1/2)*np.log(2*np.pi*prior_sd**2) - (1/2)*((p[0]-true_parameters[0])/prior_sd)**2

def log_likelihood(p):
    q_sim = simulate(p)
    # print( ss.multivariate_normal.logpdf(q,mean=q_sim,cov=np.eye(n)*sigma2) )
    # print( -(n/2)*np.log(2*np.pi*sigma2) - (1/(2*sigma2)) * (q-q_sim).T @ (q-q_sim) )
    return -(n/2)*np.log(2*np.pi*sigma2) -(1/(2*sigma2)) * (q-q_sim).T @ (q-q_sim)

def neg_log_likelihood(p):
    return (-1)*log_likelihood(p)

def log_posterior(p):
    return log_prior(p) + log_likelihood(p)

def log_kernel(pnew,pprev):
    return 0

def log_acceptance_ratio(pnew,pprev):
    log_pnew = log_posterior(pnew) + log_kernel(pnew,pprev)
    log_pold = log_posterior(pprev) + log_kernel(pprev,pnew)
    log_acc = log_pnew - log_pold
    # If exp floating point exceeded
    if log_acc >= 709: return 0, log_pnew, log_pold
    else: return log_acc, log_pnew, log_pold

# logprior = [log_prior([p]) for p in np.linspace(0.55,0.65,1000)]
# plt.figure(figsize=(10,10))
# plt.plot(np.linspace(0.55,0.65,1000),logprior)
# plt.vlines(true_parameters[0],np.min(logprior),np.max(logprior),color='red')

""" MLE computation"""
mle = fmin(neg_log_likelihood,true_parameters[0:num_learning_parameters],disp=False)[0:num_learning_parameters]
print('True parameters',true_parameters[0])
print('True parameter log likelihood',log_likelihood(true_parameters))
print('True parameter log posterior',log_posterior(true_parameters))
print('Maximum likelihood estimate',mle)
print('MLE likelihood',log_likelihood(mle))
print('MLE posterior',log_posterior(mle))

""" Data visualisation"""
plt.figure(figsize=(10,10))
plt.scatter(rho,q,color='blue')
plt.plot(rho,q_true,color='black',label='True')
plt.plot(rho,simulate([mle]),color='red',label='MLE-fitted')
plt.xlabel(r'$\rho$')
plt.ylabel(r'$q$')
plt.legend()
plt.show()


""" Space exploration"""
logprior = [log_prior([par]) for par in alphas]
plt.figure(figsize=(10,10))
plt.plot(alphas,logprior,color='red')
plt.vlines(true_parameters[0],np.min(logprior),np.max(logprior),color='black',label='True')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Log prior')
plt.legend()
plt.show()

loglike = [log_likelihood([par]) for par in alphas]
plt.figure(figsize=(10,10))
plt.plot(alphas,loglike,color='red')
plt.vlines(true_parameters[0],np.min(loglike),np.max(loglike),color='black',label='True')
plt.vlines(mle,np.min(loglike),np.max(loglike),color='red',label='MLE')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'Log likelihood')
plt.legend()
plt.show()




max_log_target = -1e9
max_log_likelihood = -1e9
max_log_likelihood_params = [-1]
if not load_data:
    # Initialise MCMC-related variables
    theta = []
    theta_proposed = []
    acc = 0

    # Store necessary parameters
    p_prev = copy.deepcopy(p0)

    print('p0',p_prev)
    print(f'Running MCMC with {N} iterations')

    # Loop through MCMC iterations
    for i in tqdm(range(N)):

        # Propose new sample
        p_new = p_prev + np.random.multivariate_normal(mean=np.zeros(num_learning_parameters),cov=K)
        #ss.multivariate_normal.rvs(mean=np.zeros(num_learning_parameters),cov=K)
        # print('p_new',p_new)
        if p_new < 0:
            print('negative p_new',p_new)
        #     p_new = - p_new

        # Calculate acceptance probability
        log_acc_ratio,lt_new,lt_prev = log_acceptance_ratio(p_new,p_prev)
        acc_ratio = min(1,np.exp(log_acc_ratio))

        ll_new = log_posterior(p_new)

        # Update maximum log-target
        if ll_new >= max_log_likelihood:
            max_log_target = lt_new
            max_log_likelihood = ll_new
            max_log_likelihood_params = p_new

        # Sample from Uniform(0,1)
        u = np.random.random()#ss.uniform.rvs(0,1)

        # Printing proposals every 0.1*Nth iteration
        if (i in [int(j/10*N) for j in range(1,11)]):
            print('p_prev',p_prev,'lt_prev',lt_prev)
            print('p_new',p_new,'lt_new',lt_new)

        # Accept/Reject
        # Compare log_alpha and log_u to accept/reject sample
        if acc_ratio >= u:
            if (i in [int(j/10*N) for j in range(1,11)]):
                print('p_new =',p_new)
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
            print('Empirical std during burnin',np.std(theta[0:burnin]))

    # Update class attributes
    theta = np.array(theta)
    theta_proposed = np.array(theta_proposed)
    print(f'Acceptance rate {int(100*acc / N)}%')

    ensure_dir(f"./data/output/debugging/")
    np.savetxt(f"./data/output/debugging/simplified_theta.txt",theta)
    np.savetxt(f"./data/output/debugging/simplified_theta_proposed.txt",theta_proposed)

else:
    theta = np.loadtxt(f"./data/output/debugging/simplified_theta.txt")
    theta_proposed = np.loadtxt(f"./data/output/debugging/simplified_theta_proposed.txt")

print('True log posterior',log_posterior(true_parameters))
print('True log likelihood',log_likelihood(true_parameters))
print('Empirical MLE',max_log_likelihood_params)
print('Empirical MLE log likelihood',log_likelihood(max_log_likelihood_params))
print('Empirical MLE log posterior',log_posterior(max_log_likelihood_params))
print('Empirical std during burnin',np.std(theta[0:burnin]))


# Reshape theta
theta = theta.reshape((N,num_learning_parameters))
theta_proposed = theta_proposed.reshape((N,num_learning_parameters))

""" Parameter posterior plots"""

# Loop through parameter indices
for par in range(num_learning_parameters):
    # Generate figure
    fig = plt.figure(figsize=(10,8))

    # Compute posterior mean
    posterior_mean = np.mean(theta[burnin:,:],axis=0)
    # Compute posterior std
    posterior_std = np.std(theta[burnin:,:],axis=0)

    prior_mean = (1/(prior_sd**2) + rho.T@rho)**(-1) * (true_parameters[0]/(prior_sd**2))
    data_mean =  (1/(prior_sd**2) + rho.T@rho)**(-1)* np.sum([q[i]*rho[i] for i in range(n)])
    true_posterior_mean = prior_mean + data_mean
    true_posterior_std = np.sqrt((1/(prior_sd**2) + rho.T@rho)**(-1))
    print(parameter_names[par])
    print('true posterior mean',true_posterior_mean)
    print('degrees of freedom',((1/(prior_sd**2) + rho.T@rho)**(-1)))
    print('true prior mean',(true_parameters[0]/(prior_sd**2)))
    print('true data mean',np.sum([q[i]*rho[i] for i in range(n)]))
    print('posterior mean',posterior_mean[par])
    print('true posterior std',true_posterior_std)
    print('posterior std',posterior_std[par])

    # Plot parameter posterior
    freq,_,_ = plt.hist(theta[burnin:,par],bins=bins,zorder=1)

    # Add labels
    plt.title(f'Parameter posterior for {parameter_names[par]} with burnin = {burnin}')
    plt.vlines(posterior_mean[par],0,np.max(freq),color='red',label=r'$\mu$',linewidth=2, zorder=3)
    plt.vlines(posterior_mean[par]-num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(posterior_mean[par]+num_stds*posterior_std[par],0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(true_parameters[par],0,np.max(freq),label='True',color='black',linewidth=2, zorder=2)
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
    plt.hlines(true_parameters[par],xmin=burnin,xmax=(theta.shape[0]),color='black',label='True',zorder=2)

    # Plot inferred mean
    plt.hlines(posterior_mean[par],xmin=burnin,xmax=(theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=3)

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

index = 0

# Create figure
fig = plt.figure(figsize=(10,8))

# Get parameters to plot
theta_subset = theta[burnin:,index]
k = len(theta_subset)
theta_proposed_subset = theta_proposed[burnin:,index]

# Add samples plot
plt.scatter(theta_subset,np.ones(k),color='red',label='Accepted',marker='x',s=50,zorder=5)
plt.scatter(theta_proposed_subset,np.ones(k),color='purple',label='Proposed',marker='x',s=50,zorder=4)

# Plot true parameters if they exist
plt.scatter(true_parameters[index],1,label='True',marker='.',s=100,color='black',zorder=7)
plt.scatter(posterior_mean[index],1,label='Mean',marker='+',s=100,color='green',zorder=8)

# Add labels
plt.xlabel(f'{parameter_names[index]}')
plt.ylabel(f'')
# Add title
plt.title(f'{parameter_names[index]} space exploration with burnin = {burnin}')
plt.yticks([])
# Add legend
plt.legend()
plt.show()
