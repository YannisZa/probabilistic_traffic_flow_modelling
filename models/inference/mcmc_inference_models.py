import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils
import numpy as np
import scipy.stats as ss
import scipy.optimize as so
import matplotlib
import matplotlib.pyplot as plt

from distutils.util import strtobool
from inference import MarkovChainMonteCarlo
from probability_distributions import *

# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

ignore_parameters = ["distribution","transformation","min_log_prob","min","max"]

class MetropolisHastings(MarkovChainMonteCarlo):
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

        # Find lower and upper bounds
        lower_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['lower_bounds'])]
        upper_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['upper_bounds'])]

        # Loop through number of parameters
        for i,k in enumerate(list(self.inference_metadata['inference']['priors'])[0:self.num_learning_parameters]):
            # Get prior hyperparameter kwargs and populate them with lower and upper parameter bounds
            hyperparams = {"plower":lower_bounds[i],"pupper":upper_bounds[i]}
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

        # Store raw hypeparameters of priors
        self.prior_hyperparameters = [[x['loc'],x['scale']] for j,x in enumerate(hyperparams_list)]#np.ones(self.num_learning_parameters))[3]

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


    def update_transition_kernel(self, proposal_stds, mcmc_type:str='vanilla_mcmc', prints:bool=False):

        # Get kernel parameters
        kernel_params = self.inference_metadata['inference'][mcmc_type]['transition_kernel']
        kernel_type = str(self.inference_metadata['inference']['transition_kernel']['type'])

        # Get number of temperature ladder steps
        temp_nsteps = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])
        # Constuct diagonal proposal distribution covariance
        K = np.diag([proposal_stds[i]**2 for i in range(len(proposal_stds))])
        # print(mcmc_type,'fixed proposal',proposal_stds)
        # Get number of iterations of MCMC
        N = max(int(self.inference_metadata['inference'][mcmc_type]['N']),1)

        # If dynamic proposal specified
        dynamic_proposal = False
        if "dynamic_proposal" in list(kernel_params.keys()):
            dynamic_proposal = bool(strtobool(kernel_params['dynamic_proposal']))

        # Get type of dynamic proposal (stochastic or deterministic)
        stochastic_proposal = False
        if "stochastic_proposal" in list(kernel_params.keys()):
            stochastic_proposal = bool(strtobool(kernel_params['stochastic_proposal']))

        # Read temperature threshold
        temperature_threshold = 0
        if "temperature_threshold" in self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']:
            temperature_threshold = float(self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']['temperature_threshold'])
        # Temperature threshold is irrelevant in a probabilistic proposal mechanism
        if dynamic_proposal and stochastic_proposal:
            temperature_threshold = 0

        # Read adaptive proposal steps for large parameter models with diffuse priors (see DeRomphs model sensitivity analysis)
        adaptive_proposal_covariances = np.ones((len(self.temperature_schedule),self.num_learning_parameters))
        adaptive_proposal_steps_flag = temperature_threshold > 0\
                                    and "proposal_stds_max" in self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']\
                                    and "proposal_stds_min" in self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']
        if adaptive_proposal_steps_flag:
            max_proposal_stds = np.array(self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']["proposal_stds_max"])
            min_proposal_stds = np.array(self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']["proposal_stds_min"])
            # Get number of temperatures above threshold
            num_temperatures = sum(self.temperature_schedule > temperature_threshold)

            # Get proposal steps for each parameter and temperature
            adaptive_proposal_steps = np.flip(np.array([np.linspace(min_proposal_stds[i],max_proposal_stds[i],num_temperatures) for i in range(self.num_learning_parameters)]),axis=1)
            # Constuct adaptive covariance - we initialise to one so that a nan does not appear in the log density
            adaptive_proposal_covariances = np.ones((len(self.temperature_schedule),self.num_learning_parameters))
            # Get threshold temperature
            threshold_temperature = next(x[0] for x in enumerate(self.temperature_schedule) if x[1] > temperature_threshold)
            # Update higher temperature proposals
            adaptive_proposal_covariances[threshold_temperature:len(self.temperature_schedule),:] = adaptive_proposal_steps.T
        print('adaptive_proposal_covariances',adaptive_proposal_covariances[10,:])

        # Read alpha step
        alpha_step = 1
        if "alpha_step" in kernel_params:
            alpha_step = float(kernel_params['alpha_step'])

        # Read method of prior sampling in dynamic kernel
        prior_sampling = 'mc' # MC, Taylor
        if dynamic_proposal and "prior_sampling" in kernel_params:
            prior_sampling = str(kernel_params['prior_sampling'])

        # Read beta distribution parameters
        a = 1.
        if "beta_dstr_a" in kernel_params:
            a = float(kernel_params['beta_dstr_a'])
        b = 10.
        if "beta_dstr_b" in kernel_params:
            b = float(kernel_params['beta_dstr_b'])

        # print('Dynamic proposal',dynamic_proposal)
        if prints:
            print(mcmc_type)
            print('Dynamic proposal',dynamic_proposal)
            print('Stochastic proposal',stochastic_proposal)
            print('Prior sampling',prior_sampling.capitalize())
            print('Proposal covariance',K)
            print('Proposal standard deviations',[np.sqrt(v) for v in np.diagonal(K)])

        # If K does not have length equal to self.num_learning_parameters raise error
        if len(np.diagonal(K)) != self.num_learning_parameters:
            raise ValueError(f'K diagonal has length {len(np.diagonal(K))} and not {self.num_learning_parameters}')

        # Get distribution corresponding to type of kernel
        distribution_sampler = utils.map_name_to_distribution_sampler(kernel_type)

        # Define proposal mechanism for Vanilla MCMC
        if mcmc_type == 'vanilla_mcmc':
            def _kernel(_pprev):
                # pnew = None
                # if self.num_learning_parameters > 1:
                # print('vanilla mcmc',np.diagonal(K))
                pnew = _pprev + distribution_sampler(np.zeros(self.num_learning_parameters),K)
                # else:
                #     pnew = pprev + distribution_sampler(0,K[0,0])
                return pnew

            def _log_kernel(p1,p2):
                return 0#multivariate_gaussian(p2,p1,K)


            # Update Vanilla MCMC kernels
            MarkovChainMonteCarlo.transition_kernel.fset(self, _kernel)
            MarkovChainMonteCarlo.log_kernel.fset(self, _log_kernel)

        # Define proposal mechanism for Thermodynamic Integration MCMC
        # Define dynamic proposal mechanism only if necessary
        if mcmc_type == 'thermodynamic_integration_mcmc':
            if dynamic_proposal:
                # If proposal is deterministic
                if not stochastic_proposal:
                    # Get second order Taylor expansion of moments of transformed prior
                    if prior_sampling.lower() == 'taylor':
                        means = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[0] for j,x in enumerate(self.prior_hyperparameters)]
                        vars = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[1] for j,x in enumerate(self.prior_hyperparameters)]
                    # Get prior mean and standard deviations
                    elif prior_sampling.lower() == 'mc':
                        means = [x[0] for x in self.prior_hyperparameters]
                        vars = [x[1]**2 for x in self.prior_hyperparameters]
                    else:
                        raise ValueError(f'Prior sampling method {prior_sampling} not recognised.')
                    # print('self.prior_hyperparameters',self.prior_hyperparameters)
                    # print('means',means)
                    # print('vars',alpha_step*vars)

                    def _dynamic_deterministic_ti_kernel(pprev,t):
                        pnew = None
                        if self.temperature_schedule[t] <= temperature_threshold:
                            if prior_sampling.lower() == 'taylor':
                                # Sample from a symmetric Normal distribution approximated from the taylor expansion of the prior's moments
                                pnew = np.random.multivariate_normal(means,alpha_step*np.diag(vars))
                            elif prior_sampling.lower() == 'mc':
                                # Sample from untransformed prior and transform sample
                                s = np.random.multivariate_normal(means,np.diag(vars))
                                pnew = self.transform_parameters(s,False)
                            # if any(np.isnan(pnew)):
                            #     print('s',s)
                            #     print('p',p)
                            #     print('pnew',pnew)
                            #     print('\n')
                            return pnew
                        else:
                            if adaptive_proposal_steps_flag:
                                # print('ti mcmc',adaptive_proposal_covariances[10,:])
                                # Sample from symmetric Gaussian kernel with adaptive proposal steps
                                # print('adaptive_proposal_covariances[t,:]**2',adaptive_proposal_covariances[t,:]**2)
                                # print('adaptive_proposal_covariances[t,:]',adaptive_proposal_covariances[t,:])
                                # if t == 29:
                                # print('Adaptive proposal',adaptive_proposal_covariances[t,:])
                                pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),np.diag(adaptive_proposal_covariances[t,:]**2))
                            else:
                                # Sample from symmetric Gaussian kernel with fixed proposal steps
                                pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),K)
                            return pnew

                    def _log_dynamic_deterministic_ti_kernel(p1,p2,t):
                        if prior_sampling.lower() == 'taylor':
                            return (self.temperature_schedule[t] <= temperature_threshold)*1 * multivariate_gaussian(p2[t,:],means,alpha_step*np.diag(vars)) + \
                                    (self.temperature_schedule[t] > temperature_threshold)*1 * ( (1-int(adaptive_proposal_steps_flag)) * multivariate_gaussian(p2[t,:],p1[t,:],K) +\
                                                                                            (int(adaptive_proposal_steps_flag)) * multivariate_gaussian(p2[t,:],p1[t,:],np.diag(adaptive_proposal_covariances[t,:]**2)) )
                        elif prior_sampling.lower() == 'mc':
                            return (self.temperature_schedule[t] <= temperature_threshold)*1 * self.evaluate_log_joint_prior(p2[t,:]) + \
                                    (self.temperature_schedule[t] > temperature_threshold)*1 * ( (1-int(adaptive_proposal_steps_flag)) * multivariate_gaussian(p2[t,:],p1[t,:],K) +\
                                                                                            (int(adaptive_proposal_steps_flag)) * multivariate_gaussian(p2[t,:],p1[t,:],np.diag(adaptive_proposal_covariances[t,:]**2)) )

                    # Update Thermodynamic Integration MCMC kernels
                    MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _dynamic_deterministic_ti_kernel)
                    MarkovChainMonteCarlo.log_ti_kernel.fset(self, _log_dynamic_deterministic_ti_kernel)

                # If proposal is stochastic/probabilistic
                else:
                    # Get second order Taylor expansion of moments of transformed prior
                    if prior_sampling.lower() == 'taylor':
                        means = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[0] for j,x in enumerate(self.prior_hyperparameters)]
                        vars = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[1] for j,x in enumerate(self.prior_hyperparameters)]
                    # Get prior mean and standard deviations
                    elif prior_sampling.lower() == 'mc':
                        means = [x[0] for x in self.prior_hyperparameters]
                        vars = [x[1]**2 for x in self.prior_hyperparameters]
                    else:
                        raise ValueError(f'Prior sampling method {prior_sampling} not recognised.')
                    # print('self.prior_hyperparameters',self.prior_hyperparameters)
                    # print('means',means)
                    # print('vars',alpha_step*vars)
                    # Compute CDF of Beta distribution for each temperature
                    beta_probabilities = ss.beta.cdf(self.temperature_schedule,a,b)

                    def _dynamic_stochastic_ti_kernel(pprev,t):
                        pnew = None
                        # u = np.random.uniform()
                        # Sample from beta distribution
                        u = np.random.beta(a,b)
                        if self.temperature_schedule[t] <= u:
                            # if self.temperature_schedule[t] == 1: print('Prior sampling')
                            if prior_sampling.lower() == 'taylor':
                                # Sample from a symmetric Normal distribution approximated from the taylor expansion of the prior's moments
                                pnew = np.random.multivariate_normal(means,alpha_step*np.diag(vars))
                            elif prior_sampling.lower() == 'mc':
                                # Sample from untransformed prior and transform sample
                                s = np.random.multivariate_normal(means,np.diag(vars))
                                pnew = self.transform_parameters(s,False)
                            # if any(np.isnan(pnew)):
                            #     print('s',s)
                            #     print('p',p)
                            #     print('pnew',pnew)
                            #     print('\n')
                            return pnew
                        else:
                            if adaptive_proposal_steps_flag:
                                # Sample from adaptive symmetric Gaussian kernel
                                pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),np.diag(adaptive_proposal_covariances[t,:]**2))
                            else:
                                # Sample from symmetric Gaussian kernel
                                pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),K)
                                return pnew

                    def _log_dynamic_stochastic_ti_kernel(p1,p2,t):
                        # return np.log( (self.temperature_schedule[t] <= 0.01)*1 * np.exp(multivariate_gaussian(p2[t,:],means,alpha_step*np.diag(vars))) + ((self.temperature_schedule[t] > 0.01)*1) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) )
                        if prior_sampling.lower() == 'taylor':
                            if adaptive_proposal_steps_flag:
                                return np.log( (1-beta_probabilities[t]) * np.exp(multivariate_gaussian(p2[t,:],means,alpha_step*np.diag(vars))) + (beta_probabilities[t]) * ( (1-int(adaptive_proposal_steps_flag)) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) +\
                                                    (int(adaptive_proposal_steps_flag)) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],np.diag(adaptive_proposal_covariances[t,:]**2))) ) )
                            else:
                                return np.log( (1-beta_probabilities[t]) * np.exp(multivariate_gaussian(p2[t,:],means,alpha_step*np.diag(vars))) + \
                                        (beta_probabilities[t]) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) )

                            #return np.log( (1 - self.temperature_schedule[t]) * np.exp(multivariate_gaussian(p2[t,:],means,alpha_step*np.diag(vars))) + self.temperature_schedule[t] * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) )
                        elif prior_sampling.lower() == 'mc':
                            # if t == 29:
                            #     print('current version',np.log( (1-beta_probabilities[t]) * np.exp(self.evaluate_log_joint_prior(p2[t,:])) + (beta_probabilities[t]) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) ))
                            #     print('truth',multivariate_gaussian(p2[t,:],p1[t,:],K) )
                            if adaptive_proposal_steps_flag:
                                return np.log( (1-beta_probabilities[t]) * np.exp(self.evaluate_log_joint_prior(p2[t,:])) + \
                                    (beta_probabilities[t]) * ( (1-int(adaptive_proposal_steps_flag)) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) +\
                                                            (int(adaptive_proposal_steps_flag)) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],np.diag(adaptive_proposal_covariances[t,:]**2))) ) )
                                #return np.log( (1 - self.temperature_schedule[t]) * np.exp(self.evaluate_log_joint_prior(p2[t,:])) + self.temperature_schedule[t] * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) )
                            else:
                                return np.log( (1-beta_probabilities[t]) * np.exp(self.evaluate_log_joint_prior(p2[t,:])) + \
                                    (beta_probabilities[t]) * np.exp(multivariate_gaussian(p2[t,:],p1[t,:],K)) )
                    # Update Thermodynamic Integration MCMC kernels
                    MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _dynamic_stochastic_ti_kernel)
                    MarkovChainMonteCarlo.log_ti_kernel.fset(self, _log_dynamic_stochastic_ti_kernel)
            else:
                def _ti_kernel(pprev,t):
                    pnew = None
                    # Sample from symmetric Gaussian kernel
                    if self.num_learning_parameters > 1:
                        pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),K)
                    else:
                        pnew = pprev + distribution_sampler(np.zeros(self.num_learning_parameters),K[0,0])

                    return pnew

                def _log_ti_kernel(p1,p2,t):
                    return 0

                # Update thermodynamic integration kernels
                MarkovChainMonteCarlo.thermodynamic_integration_transition_kernel.fset(self, _ti_kernel)
                MarkovChainMonteCarlo.log_ti_kernel.fset(self, _log_ti_kernel)


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
