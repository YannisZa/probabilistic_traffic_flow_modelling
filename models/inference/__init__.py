import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import copy
import json
import glob
import utils
import itertools
import collections.abc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tqdm import tqdm
from scipy.special import comb
from distutils.util import strtobool


# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

class MarkovChainMonteCarlo(object):

    def __init__(self,method):
        self.method = method

    def update_log_likelihood_log_pdf(self,fundamental_diagram,sigma2):
        pass

    def update_log_prior_log_pdf(self,fundamental_diagram):
        pass

    def sample_from_univariate_priors(self,num_params,N):
        pass

    def update_predictive_likelihood(self,x,y):
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
    def predictive_likelihood(self):
        return self.__predictive_likelihood

    @predictive_likelihood.setter
    def predictive_likelihood(self,predictive_likelihood):
        self.__predictive_likelihood = predictive_likelihood

    @property
    def log_unnormalised_posterior(self):
        return self.__log_unnormalised_posterior

    @log_unnormalised_posterior.setter
    def log_unnormalised_posterior(self,log_unnormalised_posterior):
        self.__log_unnormalised_posterior = log_unnormalised_posterior

    @property
    def parameter_mesh(self):
        return self.__parameter_mesh

    @parameter_mesh.setter
    def parameter_mesh(self,parameter_mesh):
        self.__parameter_mesh = parameter_mesh

    @property
    def transition_kernel(self):
        return self.__transition_kernel

    @transition_kernel.setter
    def transition_kernel(self,transition_kernel):
        self.__transition_kernel = transition_kernel

    @property
    def inference_metadata(self):
        return self.__inference_metadata

    @inference_metadata.setter
    def inference_metadata(self,inference_metadata):
        self.__inference_metadata = inference_metadata

    def update_inference_metadata(self,inference_metadata:dict={}):
        # If new metadata exist
        if bool(inference_metadata):
            self.__inference_metadata = utils.update(self.__inference_metadata, inference_metadata)

    @property
    def posterior_predictive_mean(self):
        return self.__posterior_predictive_mean

    @posterior_predictive_mean.setter
    def posterior_predictive_mean(self,posterior_predictive_mean):
        self.__posterior_predictive_mean = posterior_predictive_mean

    @property
    def posterior_predictive_std(self):
        return self.__posterior_predictive_std

    @posterior_predictive_std.setter
    def posterior_predictive_std(self,posterior_predictive_std):
        self.__posterior_predictive_std = posterior_predictive_std

    @property
    def posterior_predictive_x(self):
        return self.__posterior_predictive_x

    @posterior_predictive_x.setter
    def posterior_predictive_x(self,posterior_predictive_x):
        self.__posterior_predictive_x = posterior_predictive_x

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

    @property
    def theta(self):
        return self.__theta

    @theta.setter
    def theta(self,theta):
        self.__theta = theta

    @property
    def theta_proposed(self):
        return self.__theta_proposed

    @theta_proposed.setter
    def theta_proposed(self,theta_proposed):
        self.__theta_proposed = theta_proposed


    def setup(self,inference_params,fundamental_diagram,sigma2):
        # Update data and parameters in inference model
        self.update_data(fundamental_diagram.rho,fundamental_diagram.q,r"$\rho$",r"$q$",inference_params)

        # Update model likelihood
        self.update_log_likelihood_log_pdf(fundamental_diagram,sigma2)

        # Update model predictive likelihood
        self.update_predictive_likelihood(fundamental_diagram,sigma2)

        # Update model priors
        self.update_log_prior_log_pdf(fundamental_diagram)

        # Update model transition kernel
        self.update_transition_kernel(fundamental_diagram)


    def update_data(self,x,y,x_name,y_name,inference_params):
        self.x = x
        self.y = y
        self.x_name = x_name
        self.y_name = y_name
        self.n = y.shape[0]
        self.inference_metadata = inference_params

    def propose_new_sample(self,p):
        utils.validate_attribute_existence(self,['transition_kernel'])
        return self.__transition_kernel(p)

    def evaluate_log_joint_prior(self,p):
        utils.validate_attribute_existence(self,['log_joint_prior'])
        return self.__log_joint_prior(p)

    def evaluate_log_likelihood(self,p):
        utils.validate_attribute_existence(self,['log_likelihood'])
        return self.__log_likelihood(p)

    def evaluate_predictive_likelihood(self,p,x):
        utils.validate_attribute_existence(self,['predictive_likelihood'])
        utils.validate_attribute_existence(x,['__len__'])
        return self.__predictive_likelihood(p,x)

    def evaluate_log_posterior(self,p):
        return self.evaluate_log_likelihood(p) + self.evaluate_log_joint_prior(p)


    def vanilla_mcmc(self,fundamental_diagram,prints:bool=False,seed:int=None):
        """Vanilla MCMC method for sampling from pdf defined by log_function

        Parameters
        ----------
        fundamental_diagram: FundamentalDiagram object
            Fundamental diagram used in the likelihood
        prints : bool
            Flag for printing statements
        seed: int
            Numpy random seed

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

        # Fix random seed
        np.random.seed(seed)

        # Make sure you have stored necessary attributes
        utils.validate_attribute_existence(self,['evaluate_log_function'])

        # Make sure you have stored necessary attributes
        utils.validate_parameter_existence(['N'],self.inference_metadata['inference'])
        utils.validate_parameter_existence(['K_diagonal','beta_step'],self.inference_metadata['inference']['transition_kernel'])

        p0 = None
        # Read p0 or randomly initialise it from prior
        if utils.has_parameters(['p0'],self.inference_metadata['inference']) and not strtobool(self.inference_metadata['inference']['random_initialisation']):
            p0 = np.array(self.inference_metadata['inference']['p0'])
        else:
            p0 = self.sample_from_univariate_priors(fundamental_diagram,1)

        # Initialise output variables
        theta = []
        theta_proposed = []
        acc = 0

        # Store necessary parameters
        p_prev = p0
        # Store number of iterations
        N = int(self.inference_metadata['inference']['N'])

        if prints:
            print('p0',p_prev)
            print(f'Running MCMC with {N} iterations')

        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Evaluate log function for current sample
            lf_prev = self.__evaluate_log_function(p_prev)

            # Propose new sample
            p_new = self.propose_new_sample(p_prev)

            # Evaluate log function for proposed sample
            lf_new = self.__evaluate_log_function(p_new)

            # Printing proposals every 0.1*Nth iteration
            if prints and (i in [int(j/10*N) for j in range(1,11)]):
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
                if prints and (i in [int(j/10*N) for j in range(1,11)]):
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
                if prints and (i in [int(j/10*N) for j in range(1,11)]):
                    print('Rejected...')
                # Append to accepted and proposed sample arrays
                theta.append(p_prev)
                theta_proposed.append(p_new)

        # Update class attributes
        self.theta = np.array(theta)
        self.theta_proposed = np.array(theta_proposed)
        # Update metadata
        self.update_inference_metadata({"results":{"mcmc": {"acceptance_rate":int(100*(acc / N))}}})
        result_summary = {"results":{"mcmc":{}}}
        for i in range(self.theta.shape[1]):
            # Parameter name
            param_name = str(fundamental_diagram.parameter_names[i]).replace("$","") .replace("\\","")
            result_summary['results']['mcmc'][param_name] = {"mean":np.mean(self.theta[:,i]),"std":np.std(self.theta[:,i])}
        # Update metadata on results
        self.update_inference_metadata(result_summary)

        if prints: print(f'Acceptance rate {int(100*(acc / N))}%')

        return np.array(theta), np.array(theta_proposed), int(100*(acc / N))


    def compute_log_posterior_harmonic_mean_estimator(self,**kwargs):

        # Make sure you have stored necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])

        # Get number of MCMC iterations
        N = self.theta.shape[0]

        # ml = 0
        # print(self.theta[burnin:,:])
        # for i in range(burnin,N):
        #     term = np.exp(self.evaluate_log_likelihood(self.theta[i,:]))**(-1)
        #     print('term',term)
        #     sys.exit(1)

        # Compute log marginal likelihood
        ml = N * ( np.sum([np.exp(self.evaluate_log_likelihood(self.theta[i,:]))**(-1) for i in range(burnin,N)]) )**(-1)
        lml = np.log(ml)

        # Update metadata
        self.update_inference_metadata({"results":{"mcmc":{"log_marginal_likelihood":{"posterior_harmonic_mean":lml}}}})

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                # Print log marginal likelihood
                print(f'Log marginal likelihood = {lml}')
                # Print time execution
                toc = time.perf_counter()
                print(f"Computed posterior harmonic mean estimator in {toc - tic:0.4f} seconds")

        return lml

    def compute_gelman_rubin_statistic(self,**kwargs):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Make sure you have the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Time execution
        tic = time.perf_counter()

        # Make sure you have the necessary parameters
        utils.validate_parameter_existence(['r_critical'],self.inference_metadata['inference'])

        # Get R statistic critical value
        r_critical = float(self.inference_metadata['inference']['r_critical'])

        # Get number of chain iterations and number of chains
        n,m = self.theta.shape

        # Compute posterior mean for each parameter dimension
        posterior_parameter_means = np.array([np.mean(self.theta[:,j]) for j in range(m)])
        # Compute B
        B = n/(m-1) * np.sum([(posterior_parameter_means[j] - np.mean(self.theta,axis=(0,1)))**2 for j in range(m)])
        # Compute W
        W = (1./m) * np.sum([(1./(n-1)* np.sum([(self.theta[i,j]-posterior_parameter_means[j])**2 for i in range(n)])) for j in range(m)])
        # Compute parameter marginal posterior variance
        posterior_marginal_var = ((n-1)/n)*W + B/n
        # Compute R stastic
        r_stat = np.sqrt(posterior_marginal_var/W)

        # Decide if convergence was achieved
        if 'prints' in kwargs:
            if kwargs.get('prints'):
                # Print if chains have converged
                if r_stat < r_critical: print(r'MCMC chains have converged with $\hat{R}$=',r_stat,'!')
                else: print(r'MCMC chains have NOT converged with $\hat{R}$=',r_stat,'...')
                # Print time execution
                toc = time.perf_counter()
                print(f"Computed posterior predictive in {toc - tic:0.4f} seconds")

        # Update metadata
        self.update_inference_metadata({"results":{"mcmc":{"r_stat":r_stat}}})

        return r_stat



    """---------------------------------------------------------------------------Evaluate and generate posterior data/plots-----------------------------------------------------------------------------"""

    def evaluate_posterior_predictive_moments(self,*args,seed:int=None,**kwargs):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])

        # Set posterior predictive range to covariate range
        x = self.x
        # Set posterior predictive x based on args
        if len(args) == 1 and hasattr(args[0],'__len__'):
            x = args[0]

        # Fix random seed
        np.random.seed(seed)

        # Compute posterior predictive mean
        pp_mean = np.sum([self.evaluate_predictive_likelihood(self.theta[j,:],x) for j in range(burnin,self.theta.shape[0])],axis=0)/self.theta[burnin:].shape[0]
        # Compute posterior predictive standard deviation
        pp_std = np.sum([self.evaluate_predictive_likelihood(self.theta[j,:],x)**2 for j in range(burnin,self.theta.shape[0])],axis=0)/self.theta[burnin:].shape[0] - pp_mean**2

        # Update class variables
        self.posterior_predictive_mean = pp_mean
        self.posterior_predictive_std = pp_std
        self.posterior_predictive_x = x

        # Compute execution time
        if 'prints' in kwargs:
            if kwargs.get('prints'):
                toc = time.perf_counter()
                print(f"Computed posterior predictive in {toc - tic:0.4f} seconds")

    def evaluate_log_unnormalised_posterior(self,fundamental_diagram):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['evaluate_log_posterior'])

        # Get starting time
        start = time.time()

        parameter_ranges = []
        parameter_range_lengths = []

        # Store number of parameters
        num_params = fundamental_diagram.parameter_number
        # Store true posterior params
        true_posterior_params = self.inference_metadata['inference']['true_posterior']

        # Make sure you have enough priors
        if len(true_posterior_params.keys()) < num_params:
            raise ParameterError(f"The model has {num_params} parameter but only {len(true_posterior_params.keys())} priors were provided.")

        # Loop through number of parameters
        for k in list(true_posterior_params)[0:num_params]:
            # Define parameter range
            param_range = np.linspace(float(true_posterior_params[k]['min']),float(true_posterior_params[k]['max']),int(true_posterior_params[k]['steps']))
            # Store number of steps
            param_steps = int(true_posterior_params[k]['steps'])
            # Append to array
            parameter_ranges.append(param_range)
            parameter_range_lengths.append(param_steps)

        print(f'Evaluating a {"x".join([str(i) for i in parameter_range_lengths])} grid... Grab a cup of coffee. This will take a while...')

        # Define mesh grid
        params_mesh = np.meshgrid(*parameter_ranges[::-1])

        # Vectorize evaluate_log_function
        # evaluate_log_target_vectorized = np.vectorize(self.evaluate_log_target)#, otypes=[list])

        # Evaluate log unnormalised posterior
        log_unnormalised_posterior = np.apply_along_axis(self.evaluate_log_posterior, 0, params_mesh[::-1])

        # Reshape posterior
        log_unnormalised_posterior = log_unnormalised_posterior.reshape(tuple(parameter_range_lengths))

        # Update class attribute
        self.log_unnormalised_posterior = log_unnormalised_posterior
        self.parameter_mesh = params_mesh[::-1]

        # Print amount of time elapsed
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Log unnormalised posterior computed in {:0>2}:{:0>2}:{:05.2f} hours...".format(int(hours),int(minutes),seconds))

        return log_unnormalised_posterior,params_mesh[::-1]

    def generate_univariate_prior_plots(self,fundamental_diagram,show_plot:bool=False,prints:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['log_univariate_priors'])

        # # Create sublots
        # fig, axs = plt.subplots(figsize=(10,10*fundamental_diagram.parameter_number),nrows=fundamental_diagram.parameter_number,ncols=1)

        # Get prior distribution parameters
        prior_params = list(self.inference_metadata['inference']['priors'].values())

        figs = []
        # Loop through parameter number
        for i in range(0,fundamental_diagram.parameter_number):

            fig = plt.figure(figsize=(10,8))

            # Define x range
            xrange = np.linspace(0,10,1000)
            # Store prior hyperparameter kwargs from metadata
            hyperparams = {}
            prior_key = list(self.inference_metadata['inference']['priors'].keys())[i]
            for k, v in self.inference_metadata['inference']['priors'][prior_key].items():
                if k != "distribution":
                    hyperparams[k] = float(v)

            yrange = self.log_univariate_priors[i].pdf(xrange,**hyperparams)
            prior_mean = np.round(self.log_univariate_priors[i].mean(**hyperparams),2)
            prior_std = np.round(self.log_univariate_priors[i].std(**hyperparams),2)

            # Store distributio and parameter names
            distribution_name = prior_params[i]['distribution'].capitalize()
            parameter_name = fundamental_diagram.parameter_names[i]

            # Plot pdf
            plt.plot(xrange,yrange,color='blue',label='pdf')
            # Plot prior mean
            plt.vlines(prior_mean,ymin=-1,ymax=np.max(yrange[np.isfinite(yrange)]),color='red',label=f'mean = {prior_mean}')
            # Plot prior mean +/- prior std
            plt.hlines(np.max(yrange)/2,xmin=(prior_mean-prior_std),xmax=(prior_mean+prior_std),color='green',label=f'mean +/- std, std = {prior_std}')

            # Print hyperparameters
            if prints: print(f'Prior hypeparameters for {fundamental_diagram.parameter_names[i]}:',', '.join(['{}={!r}'.format(k, v) for k, v in hyperparams.items()]))

            # Plot true parameter if it exists
            if hasattr(fundamental_diagram,'true_parameters'):
                plt.vlines(fundamental_diagram.true_parameters[i],ymin=0,ymax=np.max(yrange[np.isfinite(yrange)]),color='black',label='true',linestyle='dashed')
            # Change x limit
            if (len(np.where((~np.isfinite(yrange)) | (yrange <= 0))[0]) > 0) and np.where((~np.isfinite(yrange)) | (yrange <= 0))[0][0] > 0:
                ximax = np.min(np.where(~np.isfinite(yrange))[0][0],np.where(yrange<= 1e-10)[0][0])
                plt.xlim(0,xrange[ximax])
            else:
                ximax = np.where(np.where(yrange <= 1e-3)[0]>=5)[0][0]
                plt.xlim(0,xrange[np.where(yrange <= 1e-3)[0][ximax]])
            # Change y limit
            plt.ylim(0,np.max(yrange[np.isfinite(yrange)])*100/99)
            # Set title
            plt.title(f"{distribution_name} prior for {parameter_name} parameter")
            # Plot legend
            plt.legend()

            # Plot figure
            if show_plot: plt.show()
            # Append to figures
            figs.append({"parameters":[fundamental_diagram.parameter_names[i]],"figure":fig})
            # Close plot
            plt.close(fig)


        return figs


    def generate_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed'])

        # Make sure posterior has right number of parameters
        if fundamental_diagram.parameter_number > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {fundamental_diagram.parameter_number}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):

            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Add samples plot
            plt.plot(range(1,self.theta[burnin:].shape[0]+1),self.theta[burnin:,p],color='blue',label='Samples')

            # Plot true parameters if they exist
            if hasattr(fundamental_diagram,'true_parameters'):
                plt.hlines(fundamental_diagram.true_parameters[p],xmin=1,xmax=(self.theta[burnin:].shape[0]),label='True',linestyle='dashed')

            # Add labels
            plt.xlabel('MCMC Iterations')
            plt.ylabel(f'MCMC Samples')
            plt.title(f'Mixing for {fundamental_diagram.parameter_names[p]} with burnin = {burnin}')
            # Add legend
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[fundamental_diagram.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs


    def generate_mcmc_acf_plots(self,fundamental_diagram,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if fundamental_diagram.parameter_number > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {fundamental_diagram.parameter_number}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])
        lags = np.min([int(self.inference_metadata['plot']['mcmc_samples']['acf_lags']),(self.theta[burnin:,:].shape[0]-1)])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):
            # Generate figure
            fig,ax = plt.subplots(1,figsize=(10,8))

            # Add ACF plot
            sm.graphics.tsa.plot_acf(self.theta[burnin:,p], ax=ax, lags=lags, title=f'ACF plot for {fundamental_diagram.parameter_names[p]} with burnin = {burnin}')

            # Add labels
            ax.set_ylabel(f'{fundamental_diagram.parameter_names[p]}')
            ax.set_xlabel('Lags')

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[fundamental_diagram.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_mcmc_parameter_posterior_plots(self,fundamental_diagram,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if fundamental_diagram.parameter_number > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {fundamental_diagram.parameter_number}')

        # Get burnin, acf lags and histogram bins from plot metadata
        burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])
        lags = np.min([int(self.inference_metadata['plot']['mcmc_samples']['acf_lags']),(self.theta[burnin:,:].shape[0]-1)])
        bins = np.max([int(self.inference_metadata['plot']['mcmc_samples']['hist_bins']),10])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):
            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Plot parameter posterior
            freq,_,_ = plt.hist(self.theta[burnin:,p],bins=bins)


            # Add labels
            plt.title(f'Parameter posterior for {fundamental_diagram.parameter_names[p]}')
            plt.vlines(np.mean(self.theta[burnin:,p]),0,np.max(freq),color='red',label=r'$\mu$', linewidth=2)
            plt.vlines((np.mean(self.theta[burnin:,p])-np.std(self.theta[burnin:,p])),0,np.max(freq),color='red',label=r'$\mu - \sigma$',linestyle='dashed', linewidth=2)
            plt.vlines((np.mean(self.theta[burnin:,p])+np.std(self.theta[burnin:,p])),0,np.max(freq),color='red',label=r'$\mu + \sigma$',linestyle='dashed', linewidth=2)
            # Plot true parameters if they exist
            if hasattr(fundamental_diagram,'true_parameters'):
                plt.vlines(fundamental_diagram.true_parameters[p],0,np.max(freq),label='True',color='black',linewidth=2)
            plt.xlabel(f'{fundamental_diagram.parameter_names[p]}')
            plt.ylabel('Sample frequency')
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[fundamental_diagram.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs


    def generate_mcmc_space_exploration_plots(self,fundamental_diagram,include_posterior:bool=False,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed'])

        # Get starting time
        start = time.time()

        # Get number of plots
        num_plots = int(comb(self.theta.shape[1],2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.parameter_number), 2))
        parameter_names = list(itertools.combinations(fundamental_diagram.parameter_names, 2))

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

            # Get burnin
            burnin = int(self.inference_metadata['plot']['mcmc_samples']['burnin'])

            # Get parameters to plot
            theta_subset = self.theta[burnin:,list(index)]
            theta_proposed_subset = self.theta_proposed[burnin:,list(index)]

            # Add samples plot
            plt.scatter(theta_subset[burnin:,index[0]],theta_subset[burnin:,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=6)
            plt.scatter(theta_proposed_subset[burnin:,index[0]],theta_proposed_subset[burnin:,index[1]],color='purple',label='Proposed',marker='x',s=50,zorder=5)

            # Get log unnormalised posterior plot
            if include_posterior:
                # Set Q_hat to log posterior
                Q_hat = self.log_unnormalised_posterior
                # Sum up dimension not plotted if there log posterior is > 2dimensional
                if len(Q_hat.shape) > 2:
                    Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.parameter_number)) - set(index))[0])

                # Try to load plot parameters
                levels = None
                # Check if all plot parameters are not empty
                if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
                    # Get number of colors in contour
                    num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),np.prod(Q_hat.shape)])
                    # Update levels
                    levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
                else:
                    vmin = np.min(Q_hat)
                    if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                        vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
                    vmax = np.max(Q_hat)
                    if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                        vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
                    if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                        # Get number of colors in contour
                        num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),np.prod(Q_hat.shape)])
                    if vmin >= vmax: print('Wrong order'); levels = np.linspace(vmax,vmin,num_colors)
                    else: levels = np.linspace(vmin,vmax,num_colors)

                # Plot countour surface
                im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels,zorder=1)
                # Plot MAP estimate
                plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=3)
                # Change limits
                plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
                plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
                # Plot colorbar
                plt.colorbar(im)
            else:
                # Get limits from plotting metadata
                plot_limits = self.inference_metadata['plot']['mcmc_samples']

                plt.xlim([float(plot_limits['xmin']),float(plot_limits['xmax'])])
                plt.ylim([float(plot_limits['ymin']),float(plot_limits['ymax'])])

            # Plot true parameters if they exist
            if hasattr(fundamental_diagram,'true_parameters'):
                plt.scatter(fundamental_diagram.true_parameters[index[0]],fundamental_diagram.true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=7)

            # Add labels
            plt.xlabel(f'{parameter_names[i][index[0]]}')
            plt.ylabel(f'{parameter_names[i][index[1]]}')
            # Add title
            plt.title(f'{parameter_names[i][index[0]]},{parameter_names[i][index[1]]} space exploration with burnin = {burnin}')
            # Add legend
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[parameter_names[i][index[0]],parameter_names[i][index[1]]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs


    def generate_log_unnormalised_posteriors_plot(self,fundamental_diagram,show_plot:bool=False):

        # Get starting time
        start = time.time()

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['log_unnormalised_posterior'])

        # Get number of plots
        num_plots = int(comb(len(self.log_unnormalised_posterior.shape),2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.parameter_number), 2))
        parameter_names = list(itertools.combinations(fundamental_diagram.parameter_names, 2))

        # Avoid plotting more than 3 plots
        if num_plots > 3:
            raise ValueError(f'Too many ({num_plots}) log posterior plots to handle!')
        elif num_plots <= 0:
            raise ValueError(f'You cannot plot {num_plots} plots!')

        # print('Generating log posterior plots')
        # Loop through each plot
        figs = []
        for i in range(num_plots):
            index = parameter_indices[i]

            # Set Q_hat to log posterior
            Q_hat = self.log_unnormalised_posterior
            # Sum up dimension not plotted if there log posterior is > 2dimensional
            if len(Q_hat.shape) > 2:
                Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.parameter_number)) - set(index))[0])

            # Try to load plot parameters
            levels = None
            # Check if all plot parameters are not empty
            if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
                # Get number of colors in contour
                num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),np.prod(Q_hat.shape)])
                # Update levels
                levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
            else:
                vmin = np.min(Q_hat)
                if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                    vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
                vmax = np.max(Q_hat)
                if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                    vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
                if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                    # Get number of colors in contour
                    num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),np.prod(Q_hat.shape)])
                levels = np.linspace(vmin,vmax,num_colors)

            # Create figure
            fig = plt.figure(figsize=(10,8))

            # Plot countour surface
            im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels)

            plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=10)
            if hasattr(fundamental_diagram,'true_parameters'):
                plt.scatter(fundamental_diagram.true_parameters[index[0]],fundamental_diagram.true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=11)
            plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
            plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
            plt.title(f'Log unnormalised posterior for {",".join(parameter_names[i])}')
            plt.xlabel(f'{parameter_names[i][index[0]]}')
            plt.ylabel(f'{parameter_names[i][index[1]]}')
            plt.colorbar(im)
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[parameter_names[i][index[0]],parameter_names[i][index[1]]],"figure":fig})
            # Close current plot
            plt.close(fig)


        return figs

    def generate_posterior_predictive_plot(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Get starting time
        start = time.time()

        figs = []

        # Create figure
        fig = plt.figure(figsize=(10,8))

        # Compute upper and lower bounds
        q_upper = self.posterior_predictive_mean + num_stds*self.posterior_predictive_std
        q_mean = self.posterior_predictive_mean
        q_lower= self.posterior_predictive_mean - num_stds*self.posterior_predictive_std

        plt.scatter(self.x,self.y,label='Observed data',color='blue',zorder=3)
        plt.plot(self.posterior_predictive_x,q_mean,color='red',label=r'$\mu$',zorder=2)
        plt.fill_between(self.posterior_predictive_x,q_upper,q_lower,alpha=0.5,color='red',label=f"$\mu$ +/- {num_stds}$\sigma$",zorder=2)
        plt.title(f"Posterior predictive for {self.inference_metadata['fundamental_diagram']} FD")
        plt.xlabel(f'{self.x_name}')
        plt.ylabel(f'{self.y_name}')
        plt.legend()

        # Show plot
        if show_plot: plt.show()
        # Append plot to list
        figs.append({"parameters":fundamental_diagram.parameter_names,"figure":fig})
        # Close current plot
        plt.close(fig)

        return figs





    """ ---------------------------------------------------------------------------Import data-----------------------------------------------------------------------------"""


    def import_metadata(self,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Make sure file exists
        if not os.path.exits((inference_filename+'_metadata.json')):
            raise FileNotFoundError(f"Metadata file {(inference_filename+'_metadata.json')} not found")

        #  Import metadata where acceptance is part of metadata
        with open((inference_filename+'_metadata.json')) as json_file:
            self.inference_metadata = json.load(json_file)

        if 'prints' in kwargs:
            if kwargs.get('prints'):print('Imported MCMC samples')


    def import_mcmc_samples(self,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Load theta from txt file
        try:
            file = (inference_filename+f'_theta.txt')
            self.theta = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob((inference_filename+f'*.txt'))))
            raise Exception(f'File {file} was not found.')

        # Load theta proposed from txt file
        try:
            file = (inference_filename+f'_theta_proposed.txt')
            self.theta_proposed = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob((inference_filename+f'*.txt'))))
            raise Exception(f'File {file} was not found.')

        if 'prints' in kwargs:
            if kwargs.get('prints'):print('Imported MCMC samples')


    def import_log_unnormalised_posterior(self,parameter_pair:list,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Get parameter names
        param_names = "_".join([str(p).replace("$","").replace("\\","") for p in parameter_pair])
        # print('Importing unnormalised posterior')
        # Load from txt file
        try:
            file = inference_filename+f'_log_unnormalised_posterior_{param_names}.txt'
            self.log_unnormalised_posterior = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "_log_unnormalised_posterior*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'_log_unnormalised_posterior_mesh_{param_names}.txt'
            self.parameter_mesh = np.loadtxt(file)
            # Define new shape
            parameter_mesh_shape = (2,self.log_unnormalised_posterior.shape[0],self.log_unnormalised_posterior.shape[1])
            # Reshape parameter mesh
            self.parameter_mesh = self.parameter_mesh.reshape(parameter_mesh_shape)
        except:
            print('Available files are',list(glob.glob(inference_filename + "_log_unnormalised_posterior_mesh*.txt")))
            raise Exception(f'File {file} was not found.')

        if 'prints' in kwargs:
            if kwargs.get('prints'): print('Imported log unnormalised posterior')

    def import_posterior_predictive(self,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Load from txt file
        try:
            file = inference_filename+f'_posterior_predictive_mean.txt'
            self.posterior_predictive_mean = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'_posterior_predictive_std.txt'
            self.posterior_predictive_std = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'_posterior_predictive_x.txt'
            self.posterior_predictive_x = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        if 'prints' in kwargs:
            if kwargs.get('prints'): print('Imported log unnormalised posterior')


    """ ---------------------------------------------------------------------------Export data/plots-----------------------------------------------------------------------------"""


    def export_log_unnormalised_posterior(self,fundamental_diagram,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['log_unnormalised_posterior','parameter_mesh','inference_metadata'])

        # Get starting time
        start = time.time()

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Export log_unnormalised_posterior
        if len(self.log_unnormalised_posterior.shape) == 2:
            # Get parameter names
            param_names = "_".join([str(k).replace("$","").replace("\\","") for k in list(self.inference_metadata['inference']['true_posterior'])[0:fundamental_diagram.parameter_number] ])
            # Save to txt file
            np.savetxt((inference_filename+f'_log_unnormalised_posterior_{param_names}.txt'),self.log_unnormalised_posterior)
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'_log_unnormalised_posterior_{param_names}.txt')}")

            # Save to txt file
            with open((inference_filename+f'_log_unnormalised_posterior_mesh_{param_names}.txt'), 'w') as outfile:
                for data_slice in self.parameter_mesh:
                    np.savetxt(outfile, data_slice, fmt='%-7.2f')
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'_log_unnormalised_posterior_mesh_{param_names}.txt')}")


        elif len(self.log_unnormalised_posterior.shape) > 2:

            raise ValueError('Not implemented yet!')
            # Get number of arrays
            num_arrays = int(comb(len(self.parameter_mesh),2))

            # # Avoid plotting more than 3 plots
            # if num_arrays > 3:
            #     raise ValueError(f'Too many ({num_plots}) log posterior arrays to export!')
            # elif num_arrays <= 0:
            #     raise ValueError(f'You cannot export {num_plots} log posterior arrays!')
            #
            # parameter_ranges = []
            # parameter_range_lengths = []
            #
            # # Store number of parameters
            # num_params = fundamental_diagram.parameter_number
            # # Store true posterior params
            # true_posterior_params = self.inference_metadata['inference']['true_posterior']
            #
            # # Make sure you have enough priors
            # if len(true_posterior_params.keys()) < num_params:
            #     raise ParameterError(f"The model has {num_params} parameter but only {len(true_posterior_params.keys())} priors were provided.")
            #
            # # Loop through number of parameters
            # for k in list(true_posterior_params)[0:num_params]:
            #     # Define parameter range
            #     param_range = np.linspace(float(true_posterior_params[k]['min']),float(true_posterior_params[k]['max']),int(true_posterior_params[k]['steps']))
            #     # Store number of steps
            #     param_steps = int(true_posterior_params[k]['steps'])
            #     # Append to array
            #     parameter_ranges.append(param_range)
            #     parameter_range_lengths.append(param_steps)
            #
            # print(f'Evaluating a {"x".join([str(i) for i in parameter_range_lengths])} grid... Grab a cup of coffee. This will take a while...')
            #
            # # Define mesh grid
            # params_mesh = np.meshgrid(*parameter_ranges[::-1])
            #
            #
            # # Get plot combinations
            # parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.parameter_number), 2))
            # parameter_names = list(itertools.combinations(fundamental_diagram.parameter_names, 2))
            #
            # # Get inference filename
            # inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])
            #
            # # Loop through each array
            # arrs = []
            # for i in range(num_arrays):
            #     index = parameter_indices[i]
            #
            #     # Set Q_hat to log posterior
            #     Q_hat = self.log_unnormalised_posterior
            #     # Sum up dimension not plotted if there log posterior is > 2dimensional
            #     Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.parameter_number)) - set(index))[0])
            #     # Get parameter names
            #     param_names = "_".join([str(p).replace("$","").replace("\\","") for p in [parameter_names[i][index[0]],parameter_names[i][index[1]]]])
            #
            #     # Save to txt file
            #     np.savetxt((inference_filename+f'_log_unnormalised_posterior_{param_names}.txt'),self.log_unnormalised_posterior)
            #     print(f"File exported to {(inference_filename+f'_log_unnormalised_posterior_{param_names}.txt')}")

        elif len(self.log_unnormalised_posterior.shape) < 2:
            raise ValueError(f'Log unnormalised posterior has shape {len(self.log_unnormalised_posterior.shape)} < 2')


    def export_samples(self,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed','inference_metadata'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Export theta
        # Save to txt file
        np.savetxt((inference_filename+'_theta.txt'),self.theta)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_theta.txt')}")

        # Export theta_proposed
        # Save to txt file
        np.savetxt((inference_filename+'_theta_proposed.txt'),self.theta_proposed)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_theta_proposed.txt')}")

    def export_posterior_predictive(self,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Export posterior_predictive_mean
        # Save to txt file
        np.savetxt((inference_filename+'_posterior_predictive_mean.txt'),self.posterior_predictive_mean)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_posterior_predictive_mean.txt')}")

        # Export posterior_predictive_std
        # Save to txt file
        np.savetxt((inference_filename+'_posterior_predictive_std.txt'),self.posterior_predictive_std)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_posterior_predictive_std.txt')}")

        # Export posterior_predictive_x
        # Save to txt file
        np.savetxt((inference_filename+'_posterior_predictive_x.txt'),self.posterior_predictive_x)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_posterior_predictive_x.txt')}")

    def export_metadata(self,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['inference_metadata'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        #  Export metadata where acceptance is part of metadata
        with open((inference_filename+'_metadata.json'), 'w') as outfile:
            json.dump(self.inference_metadata, outfile)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'_metadata.txt')}")


    def export_posterior_plots(self,figs,plot_type,**kwargs):

        # Make sure figs is not empty
        if not hasattr(figs,'__len__') or len(figs) < 1 or not all([bool(v) for v in figs]):
            raise ValueError(f'No figures found in {figs}')

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Loop through each plot and export it
        for i,f in enumerate(figs):
            # Get parameters in string format separated by _
            param_names = "_".join([str(p).replace("$","").replace("\\","") for p in figs[i]['parameters']])
            # Export plot to file
            figs[i]['figure'].savefig((inference_filename+f'_{plot_type}_{param_names}.png'),dpi=300)
            # Close plot
            plt.close(figs[i]['figure'])

            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'_{plot_type}_{param_names}.png')}")


    def export_univariate_prior_plots(self,fundamental_diagram,show_plot:bool=False,prints:bool=False):

        # Get prior plots
        fig = self.generate_univariate_prior_plots(fundamental_diagram,show_plot,prints)

        # Export them
        self.export_posterior_plots(fig,'priors')


    def export_log_unnormalised_posterior_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_log_unnormalised_posteriors_plot(fundamental_diagram,show_plot)

        # Export them
        self.export_posterior_plots(figs,"log_unnormalised_posterior")


    def export_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_mixing_plots(fundamental_diagram,show_plot)

        # Export them
        self.export_posterior_plots(figs,"mixing")

    def export_mcmc_parameter_posterior_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_parameter_posterior_plots(fundamental_diagram,show_plot)

        # Export them
        self.export_posterior_plots(figs,"parameter_posterior")


    def export_mcmc_acf_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_acf_plots(fundamental_diagram,show_plot)

        # Export them
        self.export_posterior_plots(figs,"acf")


    def export_mcmc_space_exploration_plots(self,fundamental_diagram,show_plot:bool=False):

        # Set show posterior plot to true iff the metadata says so AND you have already computed the posterior
        show_posterior = strtobool(self.inference_metadata['plot']['mcmc_samples']['include_posterior']) and utils.has_attributes(self,['log_unnormalised_posterior','parameter_mesh'])

        # Generate plots
        figs = self.generate_mcmc_space_exploration_plots(fundamental_diagram,show_posterior,show_plot)

        # Export them
        self.export_posterior_plots(figs,"space_exploration")

    def export_mcmc_posterior_predictive_plot(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Generate plots
        figs = self.generate_posterior_predictive_plot(fundamental_diagram,num_stds,show_plot)

        # Export them
        self.export_posterior_plots(figs,"posterior_predictive")
