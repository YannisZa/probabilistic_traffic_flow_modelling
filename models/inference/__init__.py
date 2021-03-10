import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import copy
import json
import utils
import itertools
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.special import comb

from tqdm import tqdm


# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

class MarkovChainMonteCarlo(object):

    def __init__(self,method):
        self.method = method

    def update_log_likelihood_log_pdf(self,fundamental_diagram,sigma2):
        pass

    def update_log_prior_log_pdf(self,num_params):
        pass

    def sample_from_univariate_priors(self,num_params,N):
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
    def log_unnormalised_posterior(self):
        return self.__log_unnormalised_posterior

    @log_unnormalised_posterior.setter
    def log_unnormalised_posterior(self,log_unnormalised_posterior):
        self.__log_unnormalised_posterior = log_unnormalised_posterior

    @property
    def log_evaluated_likelihood(self):
        return self.__log_evaluated_likelihood

    @log_evaluated_likelihood.setter
    def log_evaluated_likelihood(self,log_evaluated_likelihood):
        self.__log_evaluated_likelihood = log_evaluated_likelihood

    @property
    def transition_kernel(self):
        return self.__transition_kernel

    @transition_kernel.setter
    def transition_kernel(self,transition_kernel):
        self.__transition_kernel = transition_kernel


    @property
    def parameter_mesh(self):
        return self.__parameter_mesh

    @parameter_mesh.setter
    def parameter_mesh(self,parameter_mesh):
        self.__parameter_mesh = parameter_mesh


    @property
    def inference_metadata(self):
        return self.__inference_metadata

    @inference_metadata.setter
    def inference_metadata(self,inference_metadata):
        self.__inference_metadata = inference_metadata

    def update_inference_metadata(self,inference_metadata:dict={}):
        # If new metadata exist
        if bool(inference_metadata):
            # Add them to existing metadata
            self.__inference_metadata.update(inference_metadata)


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

    def evaluate_unnormalised_log_posterior(self,fundamental_diagram):

        # Make sure you have stored the necessary attributes
        utils.has_attributes(self,['evaluate_log_posterior'])

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

        # Evaluate log likelihood
        log_evaluated_likelihood = np.apply_along_axis(self.evaluate_log_likelihood, 0, params_mesh[::-1])
        # Reshape likelihood
        log_evaluated_likelihood = log_evaluated_likelihood.reshape(tuple(parameter_range_lengths))

        # Evaluate log prior
        log_evaluated_prior = np.apply_along_axis(self.evaluate_log_joint_prior, 0, params_mesh[::-1])
        # Reshape prior
        log_evaluated_prior = log_evaluated_prior.reshape(tuple(parameter_range_lengths))

        # Print amount of time elapsed
        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        print("Log unnormalised posterior computed in {:0>2}:{:0>2}:{:05.2f} hours...".format(int(hours),int(minutes),seconds))

        return log_unnormalised_posterior,log_evaluated_likelihood,log_evaluated_prior,params_mesh[::-1]


    def generate_unnormalised_log_posteriors_plot(self,fundamental_diagram,show_plot:bool=False):

        # Get starting time
        start = time.time()

        # Get number of plots
        num_plots = int(comb(len(self.parameter_mesh),2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.parameter_number), 2))
        parameter_names = list(itertools.combinations(fundamental_diagram.parameter_names, 2))

        # print('parameter_indices',parameter_indices)

        # Avoid plotting more than 3 plots
        if num_plots > 3:
            raise ValueError(f'Too many ({num_plots}) log posterior plots to handle!')
        elif num_plots <= 0:
            raise ValueError(f'You cannot plot {num_plots} plots!')

        # Loop through each plot
        plot_list = []
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
                # Update levels
                levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),float(self.inference_metadata['plot']['true_posterior']['num_colors']))
            elif bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                levels = np.linspace(np.min(Q_hat),np.max(Q_hat),int(self.inference_metadata['plot']['true_posterior']['num_colors']))

            # Create figure
            fig = plt.figure(figsize=(10,8))

            # Plot countour surface
            if levels is None: im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat)
            else:  im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels)

            plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='MAP',marker='x',s=200,color='blue',zorder=10)
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
            plot_list.append(fig)
            # Close current plot
            plt.close()


        return plot_list


    def generate_univariate_prior_plots(self,fundamental_diagram,show_plot:bool=False,print_statements:bool=False):

        # Make sure you have stored the necessary attributes
        utils.has_attributes(self,['log_univariate_priors'])

        # Create sublots
        fig, axs = plt.subplots(figsize=(10,10*fundamental_diagram.parameter_number),nrows=fundamental_diagram.parameter_number,ncols=1)

        # Get prior distribution parameters
        prior_params = list(self.inference_metadata['inference']['priors'].values())

        # Loop through parameter number
        for i in range(0,fundamental_diagram.parameter_number):
            # Define x range
            xrange = np.linspace(0,10,1000)
            # Store prior hyperparameter kwargs from metadata
            hyperparams = {}
            prior_key = list(self.inference_metadata['inference']['priors'].keys())[i]
            for k, v in self.inference_metadata['inference']['priors'][prior_key].items():
                if k != "prior_distribution":
                    hyperparams[k] = float(v)
            # print('hyperparams',hyperparams)
            yrange = self.log_univariate_priors[i].pdf(xrange,**hyperparams)
            prior_mean = np.round(self.log_univariate_priors[i].mean(**hyperparams),2)
            prior_std = np.round(self.log_univariate_priors[i].std(**hyperparams),2)

            # Store distributio and parameter names
            distribution_name = prior_params[i]['prior_distribution'].capitalize()
            parameter_name = fundamental_diagram.parameter_names[i]

            # Plot pdf
            axs[i].plot(xrange,yrange,color='blue',label='pdf')
            # Plot prior mean
            axs[i].vlines(prior_mean,ymin=-1,ymax=np.max(yrange[np.isfinite(yrange)]),color='red',label=f'mean = {prior_mean}')
            # Plot prior mean +/- prior std
            axs[i].hlines(np.max(yrange)/2,xmin=(prior_mean-prior_std),xmax=(prior_mean+prior_std),color='green',label=f'mean +/- std, std = {prior_std}')

            # Print hyperparameters
            if print_statements: print(f'Prior hypeparameters for {fundamental_diagram.parameter_names[i]}:',', '.join(['{}={!r}'.format(k, v) for k, v in hyperparams.items()]))

            # Plot true parameter if it exists
            if hasattr(fundamental_diagram,'true_parameters'):
                axs[i].vlines(fundamental_diagram.true_parameters[i],ymin=0,ymax=np.max(yrange[np.isfinite(yrange)]),color='black',label='true',linestyle='dashed')
            # Change x limit
            if (len(np.where((~np.isfinite(yrange)) | (yrange <= 0))[0]) > 0) and np.where((~np.isfinite(yrange)) | (yrange <= 0))[0][0] > 0:
                ximax = np.min(np.where(~np.isfinite(yrange))[0][0],np.where(yrange<= 1e-10)[0][0])
                axs[i].set_xlim(0,xrange[ximax])
            else:
                ximax = np.where(np.where(yrange <= 1e-3)[0]>=5)[0][0]
                axs[i].set_xlim(0,xrange[np.where(yrange <= 1e-3)[0][ximax]])
            # Change y limit
            axs[i].set_ylim(0,np.max(yrange[np.isfinite(yrange)])*100/99)
            # Set title
            axs[i].set_title(f"{distribution_name} prior for {parameter_name} parameter")
            # Plot legend
            axs[i].legend()

        if show_plot: plt.show()
        return fig


    def export_univariate_prior_plot(self,fundamental_diagram,data_id,show_plot:bool=False,print_statements:bool=False):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(data_id,self.method,self.inference_metadata['id'])
        # Get subplots
        fig = self.generate_univariate_prior_plots(fundamental_diagram,show_plot,print_statements)
        # Export plot to file
        fig.savefig((inference_filename+'_priors.png'),dpi=300)
        # Close plot
        plt.close(fig)
        print(f"File exported to {(inference_filename+'_priors.png')}")


    def export_unnormalised_log_posterior_plots(self,fundamental_diagram,data_id,show_plot:bool=False):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(data_id,self.method,self.inference_metadata['id'])
        # Get subplots
        figs = self.generate_unnormalised_log_posteriors_plot(fundamental_diagram,show_plot)

        # Loop through each plot and export it
        for i,f in enumerate(figs):
            # Export plot to file
            figs[i].savefig((inference_filename+f'_log_unnormalised_posterior_{i}.png'),dpi=300)
            # Close plot
            plt.close(figs[i])
            print(f"File exported to {(inference_filename+f'_log_unnormalised_posterior_{i}.png')}")


    def export_samples(self):

        # Make sure you have necessary attributes
        utils.has_attributes(self,['theta','theta_proposed','inference_metadata'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Export theta
        # Save to txt file
        np.savetxt((inference_filename+'_theta.txt'),self.theta)
        print(f"File exported to {(inference_filename+'_theta.txt')}")

        # Export theta_proposed
        # Save to txt file
        np.savetxt((inference_filename+'_theta_proposed.txt'),self.theta_proposed)
        print(f"File exported to {(inference_filename+'_theta_proposed.txt')}")


    def export_metadata(self):

        # Make sure you have necessary attributes
        utils.has_attributes(self,['inference_metadata'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        #  Export metadata where acceptance is part of metadata
        with open((inference_filename+'_metadata.json'), 'w') as outfile:
            json.dump(self.inference_metadata, outfile)
        print(f"File exported to {(inference_filename+'_metadata.txt')}")

    def export_log_unnormalised_posterior(self):

        # Make sure you have necessary attributes
        utils.has_attributes(self,['log_unnormalised_posterior','parameter_mesh'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['data_id'],self.method,self.inference_metadata['id'])

        # Export log_unnormalised_posterior
        # Save to txt file
        np.savetxt((inference_filename+'_log_unnormalised_posterior.txt'),self.log_unnormalised_posterior)
        print(f"File exported to {(inference_filename+'_log_unnormalised_posterior.txt')}")



    def setup(self,inference_params,fundamental_diagram,sigma2):
        # Update data and parameters in inference model
        self.update_data(fundamental_diagram.rho,fundamental_diagram.q,inference_params)

        # Update model likelihood
        self.update_log_likelihood_log_pdf(fundamental_diagram,sigma2)

        # Update model priors
        self.update_log_prior_log_pdf(fundamental_diagram)

        # Update model transition kernel
        self.update_transition_kernel(fundamental_diagram)


    def update_data(self,x,y,inference_params):
        self.x = x
        self.y = y
        self.n = y.shape[0]
        self.inference_metadata = inference_params

    def propose_new_sample(self,p):
        utils.has_attributes(self,['transition_kernel'])
        return self.__transition_kernel(p)

    def evaluate_log_joint_prior(self,p):
        utils.has_attributes(self,['log_joint_prior'])
        return self.__log_joint_prior(p)

    def evaluate_log_likelihood(self,p):
        utils.has_attributes(self,['log_likelihood'])
        return self.__log_likelihood(p)

    def evaluate_log_posterior(self,p):
        return self.evaluate_log_likelihood(p) + self.evaluate_log_joint_prior(p)


    # def evaluate_true_unnormalised_posterior():


    def vanilla_mcmc(self,print_stat:bool=False,seed:int=None):
        """Vanilla MCMC method for sampling from pdf defined by log_function

        Parameters
        ----------
        print_stat : bool
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
        utils.has_attributes(self,['evaluate_log_function',])

        # Make sure you have stored necessary attributes
        utils.has_parameters(['p0','N'],self.inference_metadata['inference'])
        utils.has_parameters(['K_diagonal','beta_step'],self.inference_metadata['inference']['transition_kernel'])

        # Initialise output variables
        theta = []
        theta_proposed = []
        acc = 0

        # Store necessary parameters
        p_prev = copy.deepcopy(np.array(self.inference_metadata['inference']['p0']))
        # Store number of iterations
        N = int(self.inference_metadata['inference']['N'])

        if print_stat: print('p0',p_prev)

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

        # Update class attributes
        self.theta = np.array(theta)
        self.theta_proposed = np.array(theta_proposed)
        # Update metadata
        self.update_inference_metadata({"results":{"acceptance_rate":int(100*(acc / N))}})

        print(f'Acceptance rate {int(100*(acc / N))}%')

        return np.array(theta), np.array(theta_proposed), int(100*(acc / N))


    def gelman_rubin_statistic(self):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Make sure you have the necessary attributes
        utils.has_attributes(self,['theta'])

        # Make sure you have the necessary parameters
        self.has_parameters(['r_critical'],self.inference_metadata['inference'])

        # Get R statistic critical value
        r_critical = self.inference_metadata['r_critical']

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
        if r_stat < r_critical: print(r'MCMC chains have converged with $\hat{R}$=',r_stat,'!')
        else: print(r'MCMC chains have NOT converged with $\hat{R}$=',r_stat,'...')

        # Update metadata
        self.update_inference_metadata({"results":{"r_stat":r_stat}})

        return r_stat
