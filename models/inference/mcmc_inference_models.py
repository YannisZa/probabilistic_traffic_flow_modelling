import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import numpy as np
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt

from distutils.util import strtobool
from inference import MarkovChainMonteCarlo

# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

ignore_parameters = ["distribution","transformation","min_log_prob"]

class GaussianRandomWalkMetropolisHastings(MarkovChainMonteCarlo):
    pass

    def __init__(self,inference_id):
        super().__init__(inference_id)
        MarkovChainMonteCarlo.method.fset(self, 'grwmh')
        MarkovChainMonteCarlo.log_target.fset(self, super().evaluate_log_posterior)
        MarkovChainMonteCarlo.thermodynamic_integration_log_target.fset(self, super().evaluate_thermodynamic_integration_log_posterior)

    def update_log_prior_log_pdf(self,**kwargs):

        prior_distribution_log_pdfs = []
        # Make sure you have enough priors
        if len(self.inference_metadata['inference']['priors'].keys()) < self.num_learning_parameters:
            raise ValueError(f"The model has {self.num_learning_parameters} parameter(s) but only {len(self.inference_metadata['inference']['priors'].keys())} priors were provided.")

        # Define list of hyperparameters
        hyperparams_list = []
        # Loop through number of parameters
        for k in list(self.inference_metadata['inference']['priors'])[0:self.num_learning_parameters]:
            # Get prior hyperparameter kwargs
            hyperparams = {}
            for key, v in self.inference_metadata['inference']['priors'][k].items():
                if key not in ignore_parameters:
                    hyperparams[key] = float(v)
                elif key == "transformation":
                    hyperparams[key] = v

            # Append to list
            hyperparams_list.append(hyperparams)

        # Loop through number of parameters
        for i,k in enumerate(list(self.inference_metadata['inference']['priors'])[0:self.num_learning_parameters]):
            # Define prior distribution
            prior_distr = utils.map_name_to_univariate_logpdf(self.inference_metadata['inference']['priors'][k]['distribution'])
            # Define log pdf of prior distribution
            def make_log_univariate_prior_pdf(index):
                def _log_univariate_prior_pdf(p):
                    return prior_distr(p,**hyperparams_list[index])
                return _log_univariate_prior_pdf
            # Append function to list of prior distributions
            prior_distribution_log_pdfs.append(make_log_univariate_prior_pdf(i))

        # Define multivaraite prior log pdf
        def _log_multivariate_prior_pdf(p):
            return np.sum([prior(p[j])[0] for j,prior in enumerate(prior_distribution_log_pdfs)])

        # Update super class attributes for univariate and joint priors
        MarkovChainMonteCarlo.log_joint_prior.fset(self, _log_multivariate_prior_pdf)
        MarkovChainMonteCarlo.log_univariate_priors.fset(self, prior_distribution_log_pdfs)

    def update_log_likelihood_log_pdf(self,fundamental_diagram,prints:bool=False):

        # Store distribution name
        distribution_name = self.inference_metadata['inference']['likelihood']['type'].lower()
        # Store flag for whether distribution is composed of iid observations
        iid_data = strtobool(self.simulation_metadata['error']['iid'])
        # Store flag for whether data should be kept in log scale or transformed
        multiplicative_error = strtobool(self.simulation_metadata['error']['multiplicative'])
        # Get distribution
        logpdf = utils.map_name_to_multivariate_logpdf(distribution_name,iid_data)

        # Print distribution name
        if prints: print(distribution_name,'iid data:',iid_data)

        # If data is from a simulation and sigma parameter is not learned
        if not self.learn_noise:
            # Define sigma
            # TO DO: use function to construct covariance matrix based on metadata
            if iid_data:  sigma_cov = np.eye(self.n)*self.sigma2
            else: raise ValueError('Non iid data not implemented')

            if multiplicative_error:
                def _log_likelihood(p):
                    p_transformed = self.transform_parameters(p,True)
                    return logpdf(self.log_y,loc = fundamental_diagram.log_simulate(p_transformed), scale = sigma_cov)
            else:
                print('Might have a bug - check again')
                def _log_likelihood(p):
                    p_transformed = self.transform_parameters(p,True)
                    return logpdf(np.exp(self.log_y),loc = np.exp(fundamental_diagram.log_simulate(p_transformed)), scale = sigma_cov)

        else:

            if not iid_data: raise ValueError('Non iid data not implemented')

            if multiplicative_error:
                def _log_likelihood(p):
                    p_transformed = self.transform_parameters(p,True)
                    return logpdf(self.log_y,loc = fundamental_diagram.log_simulate(p_transformed), scale = np.eye(self.n)*p_transformed[-1])
            else:
                print('Might have a bug - check again')
                def _log_likelihood(p):
                    p_transformed = self.transform_parameters(p,True)
                    return logpdf(np.exp(self.log_y),loc = np.exp(fundamental_diagram.log_simulate(p_transformed)), scale = np.eye(self.n)*p_transformed[-1])

        MarkovChainMonteCarlo.log_likelihood.fset(self, _log_likelihood)

    def update_predictive_likelihood(self,fundamental_diagram, prints:bool=False):

        # Store distribution name
        distribution_name = self.inference_metadata['inference']['likelihood']['type'].lower()

        # Make sure you use univariate distribution - and sum over each dimension
        # if distribution_name.startswith('m'):
            # distribution_name = distribution_name[1:]
        if not distribution_name.startswith('m'):
            distribution_name = 'm' + distribution_name

        # Store flag for whether data should be kept in log scale or transformed
        multiplicative_error = strtobool(self.simulation_metadata['error']['multiplicative'])
        # Store flag for whether distribution is composed of iid observations
        iid_data = strtobool(self.simulation_metadata['error']['iid'])
        # Get distribution
        distribution_sampler = utils.map_name_to_distribution_sampler(distribution_name)

        # Print distribution name
        if prints: print(distribution_name,'iid data:',iid_data)

        if not self.learn_noise:
            # Define sigma
            # TO DO: use function to construct covariance matrix based on metadata
            if iid_data:  sigma_cov = np.eye(self.n)*self.sigma2
            else: raise ValueError('Non iid data not implemented')

            if multiplicative_error:
                def _predictive_likelihood(p,x):
                    p_transformed = self.transform_parameters(p,True)
                    return distribution_sampler(mean = fundamental_diagram.log_simulate_with_x(p_transformed,x), cov = sigma_cov)
                    # return np.array([distribution_sampler(loc=(fundamental_diagram.log_simulate_with_x(p_transformed,x[i])),scale=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])
            else:
                def _predictive_likelihood(p,x):
                    p_transformed = self.transform_parameters(p,True)
                    return distribution_sampler(mean = np.exp(fundamental_diagram.log_simulate_with_x(p_transformed,x)), cov = sigma_cov)
                    # return np.array([distribution_sampler(loc=np.exp(fundamental_diagram.log_simulate_with_x(p_transformed,x[i])),scale=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])

        else:

            if not iid_data: raise ValueError('Non iid data not implemented')

            if multiplicative_error:
                def _predictive_likelihood(p,x):
                    p_transformed = self.transform_parameters(p,True)
                    return distribution_sampler(mean = fundamental_diagram.log_simulate_with_x(p_transformed,x), cov = np.eye(self.n)*p_transformed[-1])
                    # return np.array([distribution_sampler(loc=(fundamental_diagram.log_simulate_with_x(p_transformed,x[i])),scale=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])
            else:
                def _predictive_likelihood(p,x):
                    p_transformed = self.transform_parameters(p,True)
                    return distribution_sampler(mean = np.exp(fundamental_diagram.log_simulate_with_x(p_transformed,x)), cov = np.eye(self.n)*p_transformed[-1])
                    # return np.array([distribution_sampler(loc=np.exp(fundamental_diagram.log_simulate_with_x(p_transformed,x[i])),scale=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])

        MarkovChainMonteCarlo.predictive_likelihood.fset(self, _predictive_likelihood)


    def update_transition_kernel(self,mcmc_type:str='vanilla_mcmc', prints:bool=False):

        # Get kernel parameters
        kernel_params = self.inference_metadata['inference'][mcmc_type]['transition_kernel']
        kernel_type = str(kernel_params['type'])
        proposal_stds = list(kernel_params['proposal_stds'])[0:self.num_learning_parameters]
        K = np.diag([proposal_stds[i]**2 for i in range(len(proposal_stds))])
        beta_step = float(kernel_params['beta_step'])

        if prints:
            print(mcmc_type)
            print('Proposal covariance',K)
            print('Proposal standard deviations',[np.sqrt(v) for v in np.diagonal(K)])
            print('Proposal beta step',beta_step)

        # If K does not have length equal to self.num_learning_parameters raise error
        if len(np.diagonal(K)) != self.num_learning_parameters:
            raise ValueError(f'K diagonal has length {len(np.diagonal(K))} and not {self.num_learning_parameters}')

        # Get distribution corresponding to type of kernel
        distribution_sampler = utils.map_name_to_distribution_sampler(kernel_type)

        # Define proposal mechanism
        def _kernel(p):
            pnew = None
            if self.num_learning_parameters > 1:
                pnew = p + beta_step*distribution_sampler(np.zeros(self.num_learning_parameters),K)
            else:
                pnew = p + beta_step*distribution_sampler(0,K[0,0])
            return pnew

        def _log_kernel(pnew,pold):
            # TODO: define log_kernel
            return 0

        # Update super class transition kernel attribute
        if mcmc_type == 'vanilla_mcmc':
            MarkovChainMonteCarlo.transition_kernel.fset(self, _kernel)
            MarkovChainMonteCarlo.log_kernel.fset(self, _log_kernel)
        elif mcmc_type == 'thermodynamic_integration_mcmc':
            MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _kernel)
            MarkovChainMonteCarlo.log_kernel.fset(self, _log_kernel)


    def sample_from_univariate_priors(self,N):

        print('sample_from_univariate_priors needs fixing')

        # Initialise array for prior samples
        prior_samples = []

        # Make sure you have enough priors
        if len(self.inference_metadata['inference']['priors'].keys()) < self.num_learning_parameters:
            raise ValueError(f"The model has {self.num_learning_parameters} parameter(s) but only {len(self.inference_metadata['inference']['priors'].keys())} priors were provided.")

        # Loop through number of parameters
        for k in list(self.inference_metadata['inference']['priors'])[0:self.num_learning_parameters]:
            # Define prior distribution
            prior_distr = utils.map_name_to_distribution_sampler(self.inference_metadata['inference']['priors'][k]['distribution'])
            # Get prior hyperparams
            prior_hyperparams = self.inference_metadata['inference']['priors'][k]
            # Remove name of prior distribution
            prior_hyperparams.pop('distribution')
            # Convert value data type to float
            for k, v in prior_hyperparams.items():
                prior_hyperparams[k] = float(v)
            # Sample N times from univariate prior distribution
            if N == 1: prior_samples.append(prior_distr.rvs(**prior_hyperparams.values()))
            else: prior_samples.append(prior_distr.rvs(**prior_hyperparams.values(),size=N))

        return np.array(prior_samples)
