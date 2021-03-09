import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm


# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

class MarkovChainMonteCarlo(object):

    def __init__(self,name):
        self.name = name

    def update_log_likelihood_log_pdf(self,params,fd,sigma2):
        pass

    def update_log_prior_log_pdf(exp_params,num_params):
        pass

    def sample_from_univariate_priors(exp_params,num_params,N):
        pass

    @property
    def evaluate_log_function(self):
        return self.__evaluate_log_function

    @evaluate_log_function.setter
    def evaluate_log_function(self,evaluate_log_function):
        self.__evaluate_log_function = evaluate_log_function

    @property
    def log_joint_prior(self):
        return self.__log_joint_prior

    @log_joint_prior.setter
    def log_joint_prior(self,log_joint_prior):
        self.__log_joint_prior = log_joint_prior

    @property
    def log_univariate_priors(self):
        return self.__log_univariate_priors

    @log_univariate_priors.setter
    def log_univariate_priors(self,log_univariate_priors):
        self.__log_univariate_priors = log_univariate_priors

    @property
    def log_likelihood(self):
        return self.__log_likelihood

    @log_likelihood.setter
    def log_likelihood(self,log_likelihood):
        self.__log_likelihood = log_likelihood

    @property
    def transition_kernel(self):
        return self.__transition_kernel

    @transition_kernel.setter
    def transition_kernel(self,transition_kernel):
        self.__transition_kernel = transition_kernel

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self,x):
        self.__x = x

    @property
    def y(self):
        return self.__y

    @y.setter
    def y(self,y):
        self.__y = y

    @property
    def n(self):
        return self.__n

    @n.setter
    def n(self,n):
        self.__n = n

    def has_necessary_attributes(self):
        if not hasattr(self,'log_joint_prior'):
            raise AttributeError(f'Attribute log_joint_prior not found.')
        if not hasattr(self,'log_likelihood'):
            raise AttributeError(f'Attribute log_likelihood not found.')
        if not hasattr(self,'transition_kernel'):
            raise AttributeError(f'Attribute transition_kernel not found.')
        if not hasattr(self,'name'):
            raise AttributeError(f'Attribute name not found.')

    def plot_univariate_priors(self,exp_params,sim_params,fd):

        if not hasattr(self,'log_univariate_priors'):
            raise AttributeError(f'Attribute log_univariate_priors not found.')

        # Create sublots
        fig, axs = plt.subplots(figsize=(10,10*fd.parameter_number),nrows=fd.parameter_number,ncols=1)

        # Get prior distribution parameters
        prior_params = list(exp_params['inference']['priors'].values())

        # Loop through parameter number
        for i in range(0,fd.parameter_number):
            xrange = np.linspace(0.01,2,100)
            yrange = self.log_univariate_priors[i].pdf(xrange,float(prior_params[i]['a']),float(prior_params[i]['b']))

            # Store distributio and parameter names
            distribution_name = prior_params[i]['prior_distribution'].capitalize()
            parameter_name = fd.parameter_names[i]

            axs[i].plot(xrange,yrange,color='blue',label='pdf')
            axs[i].vlines(self.log_univariate_priors[i].mean(float(prior_params[i]['a']),float(prior_params[i]['b'])),ymin=-1,ymax=np.max(yrange[np.isfinite(yrange)]),color='black',label='mean')
            # plt.vlines(alpha,ymin=0,ymax=np.max(yrange[np.isfinite(yrange)]),color='black',label='true')
            if len(np.where((~np.isfinite(yrange)) | (yrange <= 0))[0]) > 0: axs[i].set_xlim(0.01,xrange[np.where((~np.isfinite(yrange)) | (yrange <= 0))[0][0]])
            axs[i].set_ylim(0,np.max(yrange[np.isfinite(yrange)]))
            axs[i].set_title(f"{distribution_name} prior for {parameter_name} parameter")
            axs[i].legend()

        plt.show()
        return axs

    def propose_new_sample(self,p):
        return self.__transition_kernel(p)

    def evaluate_log_joint_prior(self,p):
        return self.__log_joint_prior(p)

    def evaluate_log_likelihood(self,p):
        return self.__log_likelihood(p)

    def evaluate_log_target(self,p):
        return self.evaluate_log_likelihood(p) + self.evaluate_log_joint_prior(p)


    # def evaluate_true_unnormalised_posterior():


    def vanilla_mcmc(self,exp_params,print_stat:bool=False):
        """Vanilla MCMC method for sampling from pdf defined by log_function

        Parameters
        ----------
        exp_params : dict
            Multiple additional parameters that go into kernel etc.
                p0: numpy array
                    Parameter initialisation
                N: number of MCMC iterations

        print_stat : bool
            Flag for printing statements

        Returns
        -------
        numpy ndarrray
            accepted samples from target distribution
        numpy ndarrray
            accepted samples from target distribution
        float
            the proportion of accepted samples
        """

        """ Vanilla MCMC method for sampling from pdf defined by log_function
        Inputs:
            log_function - log-target distribution function
            kernel - proposal mechanism
        Returns:
            theta - accepted samples from target distribution
            theta_proposed - accepted samples from target distribution
            acc/n_iters - the proportion of accepted samples"""

        # Make sure you have stored necessary attributes
        self.has_necessary_attributes()

        # Make sure you have stored necessary attributes
        utils.has_necessary_parameters(['p0','N'],exp_params['inference'])
        utils.has_necessary_parameters(['K_diagonal','beta_step'],exp_params['inference']['transition_kernel'])

        # Initialise output variables
        theta = []
        theta_proposed = []
        acc = 0

        # Store necessary parameters
        p_prev = np.array(exp_params['inference']['p0'])
        # Store dimension of sampled variable
        n = p_prev.shape[0]
        # Store number of iterations
        N = int(exp_params['inference']['N'])

        if print_stat: print('p0',p_prev)

        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Evaluate log function for current sample
            lf_prev = self.__evaluate_log_function(p_prev)

            # Propose new sample
            p_new = self.propose_new_sample(p_prev)

            # Evaluate log function for proposed sample
            lf_new = self.__evaluate_log_function(p_new)

            # Printing proposals every 0.1*Nth iteration
            if print_stat and (i in [int(j/10*N) for j in range(1,11)]):
                print('p_prev',p_prev,'lf_prev',lf_prev)
                print('p_new',p_new,'lf_new',lf_new)
                print(f'Acceptance rate {int(100*acc / N)}%')

            # Calculate acceptance probability
            log_acc = lf_new - lf_prev
            # Sample from Uniform(0,1)
            log_u = np.log(np.random.random())


            # Accept/Reject
            # Compare log_alpha and log_u to accept/reject sample
            if min(np.exp(log_acc),1) >= np.exp(log_u):
                if print_stat and (i in [int(j/10*N) for j in range(1,11)]):
                    print('Accepted!')
                    print('p_new =',p_prev)
                # Increment accepted sample count
                acc += 1
                # Append to accepted and proposed sample arrays
                theta.append(p_new)
                theta_proposed.append(p_new)
                # Update last accepted sample
                p_prev = p_new
            else:
                if print_stat and (i in [int(j/10*N) for j in range(1,11)]):
                    print('Rejected...')
                # Append to accepted and proposed sample arrays
                theta.append(p_prev)
                theta_proposed.append(p_new)

        return np.array(theta), np.array(theta_proposed), acc / N


    def gelman_rubin_statistic(posterior_samples,**exp_params):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Make sure you have the necessary parameters
        self.has_necessary_parameters(['r_critical'],exp_params['inference'])

        # Get R statistic critical value
        r_critical = exp_params['r_critical']

        # Get number of chain iterations and number of chains
        n,m = posterior_samples.shape

        # Compute posterior mean for each parameter dimension
        posterior_parameter_means = np.array([np.mean(posterior_samples[:,j]) for j in range(m)])
        # Compute B
        B = n/(m-1) * np.sum([(posterior_parameter_means[j] - np.mean(posterior_samples,axis=(0,1)))**2 for j in range(m)])
        # Compute W
        W = (1./m) * np.sum([(1./(n-1)* np.sum([(posterior_samples[i,j]-posterior_parameter_means[j])**2 for i in range(n)])) for j in range(m)])
        # Compute parameter marginal posterior variance
        posterior_marginal_var = ((n-1)/n)*W + B/n
        # Compute R stastic
        r_stat = np.sqrt(posterior_marginal_var/W)

        # Decide if convergence was achieved
        if r_stat < r_critical: print(r'MCMC chains have converged with $\hat{R}$=',r_stat,'!')
        else: print(r'MCMC chains have NOT converged with $\hat{R}$=',r_stat,'...')

        return r_stat
