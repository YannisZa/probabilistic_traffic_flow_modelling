import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
import numpy as np
import scipy.stats as ss
import statsmodels.api as sm
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from distutils.util import strtobool
from scipy.special import loggamma

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


""" Simulation parameters """
# Fix random seed for simulating data
simulation_seed = 2021
# Define true parameters
true_parameters = [0.6,0.1,0.1]
# Define parameter names
parameter_names = [r"$\alpha$",r"$\beta$",r"$\sigma^2$"]
# Get observation noise
sigma2 = true_parameters[2]
# Rho (covariate) values
rho_min = -0.5#0.01
rho_max = 1.5#40
# Number of data points
n = 100
# Number of learning parameters
num_learning_parameters = 1

""" MCMC parameters """
# Fix random seed for running MCMC
mcmc_seed = None
# Number of MCMC iterations
N = 20000
# Diagonal covariance matrix used in proposal
K = np.diag([0.2])#0.0009845481989923365
# Initial parameters for MCMC
p0 = [0.4]#,0.2]
# Burnin for MCMC
burnin = int(N//3)

""" Miscellaneous parameters """
# Number of standard deviations to plot parameter posterior for
num_stds = 2
# Histogram number of bins
bins = 50
# Lag number of ACF plot
lags = 100
# Flag for printing
prints = True
# Flag for showing titles
show_titles = True
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
    return ss.norm.pdf(rho,loc=p[0],scale=np.sqrt(sigma2))
    #return p[0]*rho#*np.exp(-p[1]*rho)

# Simulate q
q_true = simulate(true_parameters)
q = q_true

""" Sample q from a Normal distribution """
if not load_data:
    # exp_mean = q_true / np.sqrt(1 + sigma2)
    # mean = np.log(exp_mean)
    # stdev = np.sqrt(np.log(1+sigma2))
    # q = np.array([ss.lognorm.rvs(s = stdev, loc = 0, scale = exp_mean[i]) for i in range(len(q_true))])
    # q = ss.multivariate_normal.rvs(mean=q_true,cov=sigma2*np.eye(n))
    np.savetxt(f"./data/output/debugging/simplified_uncostrained_q.txt",q)
else:
    q = np.loadtxt(f"./data/output/debugging/simplified_uncostrained_q.txt")


# Fix random seed
np.random.seed(mcmc_seed)
""" Define functions """

def log_prior(p):
    return ss.norm.logpdf(p[0],loc=true_parameters[0],scale=np.sqrt(sigma2))
    # return ss.gamma.logpdf(p[0],a=6.,scale=0.1)

def log_likelihood(p,data):
    # q_sim = simulate(p)
    return ss.norm.logpdf(p[0],loc=true_parameters[0],scale=np.sqrt(sigma2))
    # return ss.multivariate_normal.logpdf(data,mean=q_sim,cov=np.eye(n)*sigma2)

def log_posterior(p,data):
    return log_prior(p) + log_likelihood(p,data)

def log_kernel(pnew,pprev):
    return 0#ss.truncnorm.logpdf(pprev,a=0,b=1e9,loc=0,scale=np.sqrt(K[0]))

def log_acceptance_ratio(pnew,pprev,data):
    return log_posterior(pnew,data) + log_kernel(pnew,pprev) - log_posterior(pprev,data) - log_kernel(pprev,pnew)

plt.figure(figsize=(10,10))
plt.scatter(rho,q,color='blue')
plt.plot(rho,q_true,color='red')
plt.show()

print('True parameters')
for i,pname in enumerate(parameter_names):
    print(f'{pname} = {true_parameters[i]}')
print('True log_prior',log_prior(true_parameters))
print('True log_likelihood',log_likelihood(true_parameters,q))
print('True log target',log_posterior(true_parameters,q))

# samples = []
# for i in tqdm(range(10000)):
#     sam = np.array([0.6]) + ss.multivariate_normal.rvs(mean=np.zeros(num_learning_parameters),cov=K)
#     samples.append(sam[0])

# plt.figure(figsize=(10,10))
# freq1 = plt.hist(q_true,bins=50)
# plt.vlines(0.6,np.min(freq1[0]),np.max(freq1[0]),color='red',label='mu')
# plt.vlines(0.6-2*np.sqrt(K[0]),np.min(freq1[0]),np.max(freq1[0]),color='red',label='mu-2sd',linestyle='dashed')
# plt.vlines(0.6+2*np.sqrt(K[0]),np.min(freq1[0]),np.max(freq1[0]),color='red',label='mu+2sd',linestyle='dashed')
# plt.legend()
# plt.show()
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
    p_prev = p0

    print('p0',p_prev)
    print(f'Running MCMC with {N} iterations')

    # Loop through MCMC iterations
    for i in tqdm(range(N)):

        # Evaluate log function for current sample
        lt_prev = log_posterior(p_prev,q)
        ll_prev = log_likelihood(p_prev,q)

        # Propose new sample
        p_new = p_prev + ss.norm.rvs(loc=0,scale=np.sqrt(K[0]))
        #ss.truncnorm.rvs(a=0,b=1e9,loc=p_prev,scale=np.sqrt(K[0]),size=1)
        #ss.multivariate_normal.rvs(mean=np.zeros(num_learning_parameters),cov=K)

        # if p_new <= 0:
        #     print('negative proposal',p_new)
        #     p_new = -p_new

        # Evaluate log function for proposed sample
        lt_new = log_posterior(p_new,q)
        ll_new = log_likelihood(p_new,q)

        # # Make sure log acceptance ratio is finite
        # if not np.isfinite(lt_new):
        #     print('lt_new before',lt_new)
        #     if lt_new > 0: lt_new = 1e9
        #     else: lt_new = -1e9
        #     print('lt_new after',lt_new)

        # Update maximum log-target
        if ll_new >= max_log_likelihood:
            max_log_target = lt_new
            max_log_likelihood = ll_new
            max_log_likelihood_params = p_new

        # Calculate acceptance probability
        log_acc_ratio = log_acceptance_ratio(p_new,p_prev,q)
        if (log_acc_ratio) > 709: log_acc_ratio = 0
        acc_ratio = min(1,np.exp(log_acc_ratio))

        # Sample from Uniform(0,1)
        # log_u = np.log(ss.uniform.rvs(0,1))
        u = ss.uniform.rvs(0,1)


        # Printing proposals every 0.1*Nth iteration
        # if prints and (i in [int(j/10*N) for j in range(1,11)]):
        # print('i =',i,'p_prev',p_prev,'lt_prev',lt_prev,'p_new',p_new,'lt_new',lt_new)
        # print('i =',i,'log_acc_ratio',log_acc_ratio,'acc_ratio',acc_ratio,'u',u)

        # Accept/Reject
        # Compare log_alpha and log_u to accept/reject sample
        # if log_acc_ratio > log_u:
        if acc_ratio >= u:
            if prints and (i in [int(j/10*N) for j in range(1,11)]):
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
            if prints and (i in [int(j/10*N) for j in range(1,11)]):
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
    np.savetxt(f"./data/output/debugging/simplified_uncostrained_theta.txt",theta)
    np.savetxt(f"./data/output/debugging/simplified_uncostrained_theta_proposed.txt",theta_proposed)

else:
    theta = np.loadtxt(f"./data/output/debugging/simplified_uncostrained_theta.txt")
    theta_proposed = np.loadtxt(f"./data/output/debugging/simplified_uncostrained_theta_proposed.txt")

print('True log target',log_posterior(true_parameters,q))
print('True log likelihood',log_likelihood(true_parameters,q))
print('Max log likelihood',max_log_likelihood)
print('Argmax log likelihood',max_log_likelihood_params)
print('Empirical variances during burnin',np.var(theta[0:burnin]))

# Reshape theta
theta = theta.reshape((N,num_learning_parameters))
theta_proposed = theta_proposed.reshape((N,num_learning_parameters))

""" Parameter posterior plots"""

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
    # freq,_ = np.histogram(theta[burnin:,par],bins=bins)
    probs = plt.hist(theta[burnin:,par],density=True,bins=bins,zorder=1)

    # Add labels
    if show_titles: plt.title(f'Parameter posterior for {parameter_names[par]} with burnin = {burnin}')
    plt.vlines(posterior_mean[par],0,np.max(probs[0]),color='red',label=r'$\mu$',linewidth=2, zorder=3)
    plt.vlines(posterior_mean[par]-num_stds*posterior_std[par],0,np.max(probs[0]),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(posterior_mean[par]+num_stds*posterior_std[par],0,np.max(probs[0]),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
    plt.vlines(true_parameters[par],0,np.max(probs[0]),label='True',color='black',linewidth=2, zorder=2)
    plt.plot(rho,q_true,color='black',label='true distribution')
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

    print(parameter_names[par])
    print('posterior mean',np.mean(theta[burnin:,par],axis=0))

    # Plot inferred mean
    plt.hlines(np.mean(theta[burnin:,par],axis=0),xmin=burnin,xmax=(theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=3)

    # Add labels
    plt.xlabel('MCMC Iterations')
    plt.ylabel(f'MCMC Samples')
    if show_titles: plt.title(f'Mixing for {parameter_names[par]} with burnin = {burnin}')

    # Add legend
    plt.legend()

    plt.show()


""" MCMC ACF plots"""
# Loop through parameter indices
for par in range(theta.shape[1]):
    # Generate figure
    fig,ax = plt.subplots(1,figsize=(10,8))

    # Add ACF plot
    if show_titles: sm.graphics.tsa.plot_acf(theta[burnin:,par], ax=ax, lags=lags, title=f'ACF plot for {parameter_names[par]} with burnin = {burnin}')
    else: sm.graphics.tsa.plot_acf(theta[burnin:,par], ax=ax, lags=lags, title="")

    # Add labels
    ax.set_ylabel(f'{parameter_names[par]}')
    ax.set_xlabel('Lags')

    plt.show()
