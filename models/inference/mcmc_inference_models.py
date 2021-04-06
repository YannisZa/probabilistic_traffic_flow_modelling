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
        MarkovChainMonteCarlo.log_target.fset(self, super().evaluate_log_posterior)
        MarkovChainMonteCarlo.thermodynamic_integration_log_target.fset(self, super().evaluate_thermodynamic_integration_log_posterior)

    def update_log_prior_log_pdf(self,fundamental_diagram,**kwargs):

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
                if key != "distribution":
                    hyperparams[key] = float(v)
            # Append to list
            hyperparams_list.append(hyperparams)

        # Loop through number of parameters
        for i,k in enumerate(list(self.inference_metadata['inference']['priors'])[0:self.num_learning_parameters]):
            # Define prior distribution
            prior_distr = utils.map_name_to_univariate_logpdf(self.inference_metadata['inference']['priors'][k]['distribution'])
            # print('hyperparams',hyperparams)
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

    def update_log_likelihood_log_pdf(self,fundamental_diagram,**kwargs):

        # If data is from a simulation and sigma parameter is not learned
        if ((self.simulation_metadata['simulation_flag']) or not strtobool(self.inference_metadata['inference']['learn_noise'])) and (len(fundamental_diagram.true_parameters)>0):
            # Define sigma
            # TO DO: use function to construct covariance matrix based on metadata
            sigma2 = fundamental_diagram.true_parameters[-1]
            sigma_cov = np.eye(self.n)*fundamental_diagram.true_parameters[-1]

            # Store distribution name
            distribution_name = self.inference_metadata['inference']['likelihood']['type'].lower()
            # Store flag for whether distribution is composed of iid observations
            iid_data = strtobool(self.simulation_metadata['error']['iid'])
            # Store flag for whether model is additive or multiplicative
            multiplicative_model = strtobool(self.simulation_metadata['error']['multiplicative'])
            # Store flag and type of parameter transformation
            transformed_parameters, parameter_transformation, parameter_inverse_transformation = utils.map_name_to_parameter_transformation(self.inference_metadata['parameter_transformation'])
            # Get distribution
            logpdf = utils.map_name_to_multivariate_logpdf(distribution_name,iid_data)
            # Print distribution name
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(distribution_name,'iid data:',iid_data)

            if multiplicative_model:
                def _log_likelihood(p):
                    p_transformed = parameter_inverse_transformation(p)
                    return logpdf(self.log_y,loc = fundamental_diagram.log_simulate(p_transformed), scale = sigma_cov)
            else:
                def _log_likelihood(p):
                    p_transformed = parameter_inverse_transformation(p)
                    return logpdf(np.exp(self.log_y),loc = np.exp(fundamental_diagram.log_simulate(p_transformed)), scale = sigma_cov)

            # if self.inference_metadata['inference']['likelihood']['type'].lower() in ['mlognormal','lognormal']:
            #     if 'prints' in kwargs:
            #         if kwargs.get('prints'): print('multivariate lognormal')
            #     print('multivariate lognormal')
            #     def _log_likelihood(p):
            #         q = fundamental_diagram.simulate(p)
            #         exp_mean = q / np.sqrt(1 + sigma2)
            #         mean = np.log(exp_mean)
            #         var = np.log(1 + sigma2)
            #         stdev = np.sqrt(var)
            #         # print(np.sum([ss.lognorm.logpdf(self.log_y[i],s=stdev,loc=0,scale=exp_mean[i]) for i in range(len(self.log_y))]))
            #         # print(-self.n*np.log(stdev) - (self.n/2)*np.log(2*np.pi) - np.sum(np.log(self.log_y)) - (1/(2*var)) * np.sum([(np.log(self.log_y[i])-mean[i])**2 for i in range(self.n)]))
            #         # print((ss.multivariate_normal.logpdf(np.log(self.log_y),mean=mean,cov=np.eye(self.n)*var) - np.sum(np.log(self.log_y))))
            #         return((ss.multivariate_normal.logpdf(np.log(self.log_y),mean=mean,cov=np.eye(self.n)*var) - np.sum(np.log(self.log_y))))
        else:
            raise ValueError('Not implemented yet')

            if self.inference_metadata['inference']['likelihood']['type'].lower() in ['mnormal','normal']:
                def _log_likelihood(p):
                    return ss.multivariate_normal.logpdf(self.log_y,fundamental_diagram.simulate(p),np.eye(self.n)*p[-1])
            if self.inference_metadata['inference']['likelihood']['type'].lower() in ['mlognormal','lognormal']:
                def _log_likelihood(p):
                    return np.sum([ss.lognorm.logpdf(x=self.log_y[i],s=np.sqrt(p[-1]),scale=fundamental_diagram.simulate(p)[i]) for i in range(self.n)])

        MarkovChainMonteCarlo.log_likelihood.fset(self, _log_likelihood)

    def update_predictive_likelihood(self,fundamental_diagram, **kwargs):

        # Get likelihood numpy sampler from metadata
        likelihood_sampler = utils.map_name_to_distribution_sampler(self.inference_metadata['inference']['likelihood']['type'].lower())

        if ((fundamental_diagram.simulation_flag) or not strtobool(self.inference_metadata['inference']['learn_noise'])) and (len(fundamental_diagram.true_parameters) > 0):

            if self.n > 1 and self.inference_metadata['inference']['likelihood']['type'].lower() in ['lognormal','normal']:
                raise ValueError(f"Univariate {self.inference_metadata['inference']['likelihood']['type']} likelihood does not work with {self.n}-dimensional data.")

            # Define sigma
            sigma_cov = np.eye(self.n)*fundamental_diagram.true_parameters[-1]

            print('Needs fixing')
            # raise ValueError('Needs fixing')

            if 'lognormal' in self.inference_metadata['inference']['likelihood']['type'].lower() and self.n >= 1:
                if 'prints' in kwargs:
                    if kwargs.get('prints'): print('lognormal predictive likelihood')
                def _predictive_likelihood(p:list,x:list):
                    # Change the code below
                    # This computation could be made using np.random.multivariate_normal
                    return np.array([likelihood_sampler(mean=(np.log(fundamental_diagram.simulate_with_x(p,x[i]))),sigma=np.sqrt(sigma_cov[i,i])) for i in range(len(x))])
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
        proposal_stds = list(kernel_params['proposal_stds'])
        K_diagonal = np.diag([proposal_stds[i]**2 for i in range(len(proposal_stds))])
        beta_step = float(kernel_params['beta_step'])

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                print(mcmc_type)
                print('Proposal covariance',K_diagonal)
                print('Proposal standard deviations',[np.sqrt(v) for v in np.diagonal(K_diagonal)])
                print('Proposal beta step',beta_step)

        # If K_diagonal does not have length equal to self.num_learning_parameters raise error
        if len(K_diagonal) != self.num_learning_parameters:
            raise LengthError(f'K_diagonal has length {len(K_diagonal)} and not {self.num_learning_parameters}')


        # Get distribution corresponding to type of kernel
        distribution_sampler = utils.map_name_to_distribution_sampler(kernel_type)

        # Define proposal mechanism
        def _kernel(p):
            pnew = None
            if self.num_learning_parameters > 1:
                pnew = p + beta_step*distribution_sampler.rvs(np.zeros(self.num_learning_parameters),np.diag(K_diagonal))
            else:
                pnew = p + beta_step*distribution_sampler.rvs(0,K_diagonal[0])
            return pnew


        # Update super class transition kernel attribute
        if mcmc_type == 'vanilla_mcmc':
            MarkovChainMonteCarlo.transition_kernel.fset(self, _kernel)
        elif mcmc_type == 'thermodynamic_integration_mcmc':
            MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _kernel)


    def sample_from_univariate_priors(self,fundamental_diagram,N):
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
