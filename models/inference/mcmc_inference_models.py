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

class GaussianRandomWalkMetropolisHastings(MarkovChainMonteCarlo):
    pass

    def __init__(self,inference_id):
        super().__init__(inference_id)
        MarkovChainMonteCarlo.method.fset(self, 'grwmh')
        MarkovChainMonteCarlo.evaluate_log_target.fset(self, super().evaluate_log_posterior)
        MarkovChainMonteCarlo.evaluate_thermodynamic_integration_log_target.fset(self, super().evaluate_thermodynamic_integration_log_posterior)

    def update_log_prior_log_pdf(self,fundamental_diagram,**kwargs):

        prior_distribution_log_pdfs = []
        prior_distributions = []

        # Store number of parameters
        num_params = fundamental_diagram.num_learning_parameters

        # Make sure you have enough priors
        if len(self.inference_metadata['inference']['priors'].keys()) < num_params:
            raise ParameterError(f"The model has {num_params} parameter but only {len(self.inference_metadata['inference']['priors'].keys())} priors were provided.")

        # Define list of hyperparameters
        hyperparams_list = []
        # Loop through number of parameters
        for k in list(self.inference_metadata['inference']['priors'])[0:num_params]:
            # Get prior hyperparameter kwargs
            hyperparams = {}
            for key, v in self.inference_metadata['inference']['priors'][k].items():
                if key != "distribution":
                    hyperparams[key] = float(v)
            # Append to list
            hyperparams_list.append(hyperparams)

        # Loop through number of parameters
        for i,k in enumerate(list(self.inference_metadata['inference']['priors'])[0:num_params]):
            # Define prior distribution
            prior_distr = utils.map_name_to_scipy_distribution(self.inference_metadata['inference']['priors'][k]['distribution'])

            # print('hyperparams',hyperparams)
            # Define log pdf of prior distribution
            def make_log_univariate_prior_pdf(index):
                def _log_univariate_prior_pdf(p):
                    return prior_distr.logpdf(p,**hyperparams_list[index])
                return _log_univariate_prior_pdf
            # Append function to list of prior distributions
            prior_distribution_log_pdfs.append(make_log_univariate_prior_pdf(i))
            prior_distributions.append(prior_distr)

        # Define multivaraite prior log pdf
        def _log_multivariate_prior_pdf(p):
            return np.sum([prior(p[j]) for j,prior in enumerate(prior_distribution_log_pdfs)])

        # Update super class attributes for univariate and joint priors
        MarkovChainMonteCarlo.log_joint_prior.fset(self, _log_multivariate_prior_pdf)
        MarkovChainMonteCarlo.log_univariate_priors.fset(self, prior_distributions)

    def update_log_likelihood_log_pdf(self,fundamental_diagram,**kwargs):

        # If data is from a simulation and sigma parameter is not learned
        if ((fundamental_diagram.simulation_flag) or not strtobool(self.inference_metadata['inference']['learn_noise'])) and (len(fundamental_diagram.true_parameters)>0):
            # Define sigma
            sigma_cov = np.eye(self.n)*fundamental_diagram.true_parameters[-1]
            if self.inference_metadata['inference']['likelihood']['type'] in ['mnormal','normal']:
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('multivariate normal')
                def _log_likelihood(p):
                    return ss.multivariate_normal.logpdf(self.y,fundamental_diagram.simulate(p),sigma_cov)
            if self.inference_metadata['inference']['likelihood']['type'] in ['mlognormal','lognormal']:
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('multivariate lognormal')
                def _log_likelihood(p):
                    return (ss.multivariate_normal.logpdf(np.log(self.y),np.log(fundamental_diagram.simulate(p)),sigma_cov) - np.sum(np.log(self.y)))
        else:
            raise ValueError('Not implemented yet')

            if self.inference_metadata['inference']['likelihood']['type'] in ['mnormal','normal']:
                def _log_likelihood(p):
                    return ss.multivariate_normal.logpdf(self.y,fundamental_diagram.simulate(p),np.eye(self.n)*p[-1])
            if self.inference_metadata['inference']['likelihood']['type'] in ['mlognormal','lognormal']:
                def _log_likelihood(p):
                    return np.sum([ss.lognorm.logpdf(x=self.y[i],s=np.sqrt(p[-1]),scale=fundamental_diagram.simulate(p)[i]) for i in range(self.n)])

        MarkovChainMonteCarlo.log_likelihood.fset(self, _log_likelihood)

    def update_predictive_likelihood(self,fundamental_diagram, **kwargs):

        # Get likelihood numpy sampler from metadata
        likelihood_sampler = utils.map_name_to_numpy_distribution(self.inference_metadata['inference']['likelihood']['type'])

        if ((fundamental_diagram.simulation_flag) or not strtobool(self.inference_metadata['inference']['learn_noise'])) and (len(fundamental_diagram.true_parameters) > 0):

            if self.n > 1 and self.inference_metadata['inference']['likelihood']['type'] in ['lognormal','normal']:
                raise ValueError(f"Univariate {self.inference_metadata['inference']['likelihood']['type']} likelihood does not work with {self.n}-dimensional data.")

            # Define sigma
            sigma_cov = np.eye(self.n)*fundamental_diagram.true_parameters[-1]

            if 'lognormal' in self.inference_metadata['inference']['likelihood']['type'] and self.n >= 1:
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('lognormal predictive likelihood')
                def _predictive_likelihood(p:list,x:list):
                    # This computation could be made using np.random.multivariate_normal
                    return np.array([likelihood_sampler(mean=np.log(fundamental_diagram.simulate_with_x(p,x[i])),sigma=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])
            elif self.n == 1:#self.inference_metadata['inference']['likelihood']['type'] == 'normal':
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('univariate normal predictive likelihood')
                def _predictive_likelihood(p:list,x:list):
                    return np.array([likelihood_sampler(fundamental_diagram.simulate_with_x(p,x[i]),np.sqrt(sigma_cov[i,i])) for i in range(len(x))])
            elif self.n > 1:
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('multivariate normal predictive likelihood')
                def _predictive_likelihood(p:list,x:list):
                    return np.array([likelihood_sampler(fundamental_diagram.simulate_with_x(p,x[i]),sigma_cov) for i in range(len(x))])
            else:
                raise ValueError(f'Number of data points {self.n} are too few...')
        else:
            raise ValueError('Not implemented yet')

            if 'lognormal' in self.inference_metadata['inference']['likelihood']['type'] and self.n >= 1:
                def _predictive_likelihood(p:list,x:list):
                    return np.array([likelihood_sampler(mean=np.log(fundamental_diagram.simulate_with_x(p,x[i])),sigma=np.sqrt(p[-1])) for i in range(len(x))])
            elif self.n == 1:#self.inference_metadata['inference']['likelihood']['type'] == 'normal':
                def _predictive_likelihood(p:list,x:list):
                    return np.array([likelihood_sampler(fundamental_diagram.simulate_with_x(p,x[i]),np.sqrt(p[-1])) for i in range(len(x))])
            elif self.n > 1:
                def _predictive_likelihood(p:list,x:list):
                    return np.array([likelihood_sampler(fundamental_diagram.simulate_with_x(p,x[i]),np.eye(self.n)*p[-1]) for i in range(len(x))])
            else:
                raise ValueError(f'Number of data points {self.n} are too few...')

        MarkovChainMonteCarlo.predictive_likelihood.fset(self, _predictive_likelihood)


    def update_transition_kernel(self,fundamental_diagram,mcmc_type:str='vanilla_mcmc', **kwargs):

        # Get kernel parameters
        kernel_params = self.inference_metadata['inference'][mcmc_type]['transition_kernel']
        kernel_type = str(kernel_params['type'])
        K_diagonal = list(kernel_params['K_diagonal'])
        beta_step = float(kernel_params['beta_step'])

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                print(mcmc_type)
                print('Proposal var',K_diagonal)
                print('Proposal beta step',beta_step)

        # Store number of parameters
        num_params = fundamental_diagram.num_learning_parameters

        # If K_diagonal does not have length equal to num_params raise error
        if len(K_diagonal) != num_params:
            raise LengthError(f'K_diagonal has length {len(K_diagonal)} and not {num_params}')

        # Depending on kernel type define proposal mechanism
        if kernel_type.lower() in ['normal','mnormal','gaussian','mgaussian']:
            # Get distribution corresponding to type of kernel
            kernel_distr = utils.map_name_to_numpy_distribution(kernel_type)

            if not strtobool(self.inference_metadata['inference'][mcmc_type]['transition_kernel']['constrained']):
                # Define Gaussian random walk proposal mechanism
                def _kernel(p):
                    pnew = None
                    if num_params > 1:
                        pnew = p + beta_step*kernel_distr(np.zeros(num_params),np.diag(K_diagonal))
                    else:
                        pnew = p + beta_step*kernel_distr(0,K_diagonal[0])

            elif self.inference_metadata['inference'][mcmc_type]['transition_kernel']['action'] == 'reflect':
                # Define Gaussian random walk proposal mechanism
                def _kernel(p):
                    pnew = None
                    if num_params > 1:
                        pnew = p + beta_step*kernel_distr(np.zeros(num_params),np.diag(K_diagonal))
                    else:
                        pnew = p + beta_step*kernel_distr(0,K_diagonal[0])

                    return self.reflect_proposal(pnew,mcmc_type)

        # Update super class transition kernel attribute
        if mcmc_type == 'vanilla_mcmc':
            MarkovChainMonteCarlo.transition_kernel.fset(self, _kernel)
        elif mcmc_type == 'thermodynamic_integration_mcmc':
            MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _kernel)


    def sample_from_univariate_priors(self,fundamental_diagram,N):
        # Initialise array for prior samples
        prior_samples = []

        # Get number of parameters
        num_params = fundamental_diagram.num_learning_parameters

        # Make sure you have enough priors
        if len(self.inference_metadata['inference']['priors'].keys()) < num_params:
            raise ParameterError(f"The model has {num_params} parameter but only {len(self.inference_metadata['inference']['priors'].keys())} priors were provided.")

        # Loop through number of parameters
        for k in list(self.inference_metadata['inference']['priors'])[0:num_params]:
            # Define prior distribution
            prior_distr = utils.map_name_to_numpy_distribution(self.inference_metadata['inference']['priors'][k]['distribution'])
            # Get prior hyperparams
            prior_hyperparams = self.inference_metadata['inference']['priors'][k]
            # Remove name of prior distribution
            prior_hyperparams.pop('distribution')
            # Convert value data type to float
            for k, v in prior_hyperparams.items():
                prior_hyperparams[k] = float(v)
            # Sample N times from univariate prior distribution
            if N == 1: prior_samples.append(prior_distr(*prior_hyperparams.values()))
            else: prior_samples.append(prior_distr(*prior_hyperparams.values(),size=N))

        return np.array(prior_samples)
