import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import numpy as np
import scipy.stats as ss
import matplotlib
import matplotlib.pyplot as plt

from inference import MarkovChainMonteCarlo

# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

class GaussianRandomWalkMetropolisHastings(MarkovChainMonteCarlo):
    pass

    def __init__(self):
        super().__init__('grwmh')
        MarkovChainMonteCarlo.evaluate_log_function.fset(self, super().evaluate_log_posterior)

    def update_log_likelihood_log_pdf(self,fundamental_diagram,sigma2):

        # Define covariance
        sigma_cov = None
        if hasattr(sigma2,'__len__'):
            sigma_cov = np.diag(sigma2)
        elif sigma2 is not None:
            sigma_cov = np.eye(self.n)*sigma2

        if (bool(self.inference_metadata['simulation_flag']) or not bool(self.inference_metadata['inference']['learn_noise'])) and (sigma_cov is not None):
            if self.inference_metadata['inference']['likelihood']['type'] in ['mnormal','normal']:
                def _log_likelihood(p):
                    return ss.multivariate_normal.logpdf(self.y,fundamental_diagram.simulate(p),sigma_cov)
            if self.inference_metadata['inference']['likelihood']['type'] in ['mlognormal','lognormal']:
                def _log_likelihood(p):
                    return np.sum([ss.lognorm.logpdf(x=self.y[i],s=np.sqrt(sigma_cov[i,i]),scale=fundamental_diagram.simulate(p)[i]) for i in range(self.n)])
        else:
            if self.inference_metadata['inference']['likelihood']['type'] in ['mnormal','normal']:
                def _log_likelihood(p):
                    return ss.multivariate_normal.logpdf(self.y,fundamental_diagram.simulate(p),np.eye(self.n)*p[-1])
            if self.inference_metadata['inference']['likelihood']['type'] in ['mlognormal','lognormal']:
                def _log_likelihood(p):
                    return np.sum([ss.lognorm.logpdf(x=self.y[i],s=np.sqrt(sigma_cov[i,i]),scale=fundamental_diagram.simulate(p)[i]) for i in range(self.n)])

        MarkovChainMonteCarlo.log_likelihood.fset(self, _log_likelihood)

    def update_log_prior_log_pdf(self,fundamental_diagram):

        prior_distribution_log_pdfs = []
        prior_distributions = []

        # Store number of parameters
        num_params = fundamental_diagram.parameter_number

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
                if key != "prior_distribution":
                    hyperparams[key] = float(v)
            # Append to list
            hyperparams_list.append(hyperparams)

        # Loop through number of parameters
        for i,k in enumerate(list(self.inference_metadata['inference']['priors'])[0:num_params]):
            # Define prior distribution
            prior_distr = utils.map_name_to_scipy_distribution(self.inference_metadata['inference']['priors'][k]['prior_distribution'])

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


    def update_transition_kernel(self,fundamental_diagram):

        # Get kernel parameters
        kernel_params = self.inference_metadata['inference']['transition_kernel']
        kernel_type = str(kernel_params['type'])
        K_diagonal = list(kernel_params['K_diagonal'])
        beta_step = float(kernel_params['beta_step'])

        # Store number of parameters
        num_params = fundamental_diagram.parameter_number

        # If K_diagonal does not have length equal to num_params raise error
        if len(K_diagonal) != num_params:
            raise LengthError(f'K_diagonal has length {len(K_diagonal)} and not {num_params}')

        # Depending on kernel type define proposal mechanism
        if kernel_type in ['normal','mnormal']:
            # Get distribution corresponding to type of kernel
            kernel_distr = utils.map_name_to_numpy_distribution(kernel_type)

            # Define Gaussian random walk proposal mechanism
            def _kernel(p):
                if num_params > 1: return p + beta_step*kernel_distr(np.zeros(num_params),np.diag(K_diagonal))
                else: return p + beta_step*kernel_distr(0,K_diagonal[0])


        # Update super class transition kernel attribute
        MarkovChainMonteCarlo.transition_kernel.fset(self, _kernel)


    def sample_from_univariate_priors(self,num_params,N):
        # Initialise array for prior samples
        prior_samples = []

        # Make sure you have enough priors
        if len(self.inference_metadata['inference']['priors'].keys()) < num_params:
            raise ParameterError(f"The model has {num_params} parameter but only {len(self.inference_metadata['inference']['priors'].keys())} priors were provided.")

        # Loop through number of parameters
        for k in list(self.inference_metadata['inference']['priors'])[0:num_params]:
            # Define prior distribution
            prior_distr = utils.map_name_to_numpy_distribution(self.inference_metadata['inference']['priors'][k]['prior_distribution'])
            # Sample N times from univariate prior distribution
            if N == 1: prior_samples.append(prior_distr(float(self.inference_metadata['inference']['priors'][k]['a']),float(self.inference_metadata['inference']['priors'][k]['b'])))
            else: prior_samples.append(prior_distr(float(self.inference_metadata['inference']['priors'][k]['a']),float(self.inference_metadata['inference']['priors'][k]['b']),N))

        return np.array(prior_samples)
