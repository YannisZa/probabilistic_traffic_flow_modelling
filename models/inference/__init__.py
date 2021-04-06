import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import time
import copy
import json
import glob
import utils
import random
import warnings
import itertools
import collections.abc
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import statsmodels.api as sm

from tqdm import tqdm
from scipy.special import comb
from scipy.optimize import fmin
from distutils.util import strtobool
from . import temperature_schedules as tp

# matplotlib settings
matplotlib.rc('font', **{'size' : 18})

class MarkovChainMonteCarlo(object):

    def __init__(self,inference_id):
        self.inference_id = inference_id

    def update_log_likelihood_log_pdf(self,fundamental_diagram,sigma2):
        pass

    def update_log_prior_log_pdf(self,fundamental_diagram):
        pass

    def sample_from_univariate_priors(self,num_params,N):
        pass

    def update_predictive_likelihood(self,x,y):
        pass

    @property
    def log_target(self):
        return self.__log_target

    @log_target.setter
    def log_target(self,log_target):
        self.__log_target = log_target

    @property
    def thermodynamic_integration_log_target(self):
        return self.__thermodynamic_integration_log_target

    @thermodynamic_integration_log_target.setter
    def thermodynamic_integration_log_target(self,thermodynamic_integration_log_target):
        self.__thermodynamic_integration_log_target = thermodynamic_integration_log_target

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
    def temperature_schedule(self):
        return self.__temperature_schedule

    @temperature_schedule.setter
    def temperature_schedule(self,temperature_schedule):
        self.__temperature_schedule = temperature_schedule

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
    def thermodynamic_integration_transition_kernel(self):
        return self.__thermodynamic_integration_transition_kernel

    @thermodynamic_integration_transition_kernel.setter
    def thermodynamic_integration_transition_kernel(self,thermodynamic_integration_transition_kernel):
        self.__thermodynamic_integration_transition_kernel = thermodynamic_integration_transition_kernel

    @property
    def inference_metadata(self):
        return self.__inference_metadata

    @inference_metadata.setter
    def inference_metadata(self,inference_metadata):
        self.__inference_metadata = inference_metadata

    @property
    def simulation_metadata(self):
        return self.__simulation_metadata

    @simulation_metadata.setter
    def simulation_metadata(self,simulation_metadata):
        self.__simulation_metadata = simulation_metadata

    @property
    def method(self):
        return self.__method

    @method.setter
    def method(self,method):
        self.__method = method

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
    def num_learning_parameters(self):
        return self.__num_learning_parameters

    @num_learning_parameters.setter
    def num_learning_parameters(self,num_learning_parameters):
        self.__num_learning_parameters = num_learning_parameters

    @property
    def x(self):
        return self.__x

    @x.setter
    def x(self,x):
        self.__x = x

    @property
    def log_y(self):
        return self.__log_y

    @log_y.setter
    def log_y(self,log_y):
        self.__log_y = log_y

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

    @property
    def thermodynamic_integration_theta(self):
        return self.__thermodynamic_integration_theta

    @thermodynamic_integration_theta.setter
    def thermodynamic_integration_theta(self,thermodynamic_integration_theta):
        self.__thermodynamic_integration_theta = thermodynamic_integration_theta


    def populate(self,fundamental_diagram):

        # Decide how many parameters to learn
        if not (self.simulation_metadata['simulation_flag']):
            print('Raw data (non-simulation)')
            self.num_learning_parameters = len(fundamental_diagram.simulation_metadata['true_parameters'].keys())-1
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]
        elif (not strtobool(self.inference_metadata['learn_noise'])) and (len(fundamental_diagram.true_parameters)>0):
            print('Simulation without sigma learning')
            self.num_learning_parameters = len(fundamental_diagram.simulation_metadata['true_parameters'].keys())-1
            self.true_parameters = fundamental_diagram.true_parameters
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]
        else:
            print('Simulation with sigma learning')
            self.num_learning_parameters = len(fundamental_diagram.simulation_metadata['true_parameters'].keys())
            self.true_parameters = fundamental_diagram.true_parameters
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]

        print('Number of learning parameters',self.num_learning_parameters)

        # Update data and parameters in inference model
        self.update_data(fundamental_diagram.rho,fundamental_diagram.log_q,r"$\rho$",r"$\log q$")

        # Update temperature schedule
        self.update_temperature_schedule()

        # Update model likelihood
        self.update_log_likelihood_log_pdf(fundamental_diagram)

        # Update model predictive likelihood
        self.update_predictive_likelihood(fundamental_diagram)

        # Update model priors
        self.update_log_prior_log_pdf(fundamental_diagram)

        # Update model transition kernels for vanilla and thermodynamic integration MCMC
        self.update_transition_kernel(fundamental_diagram,'vanilla_mcmc')
        self.update_transition_kernel(fundamental_diagram,'thermodynamic_integration_mcmc')

        # Add results key to metadata
        self.__inference_metadata['results'] = {}


    def update_data(self,x,y,x_name,y_name):
        self.x = x
        self.log_y = y
        self.x_name = x_name
        self.log_y_name = y_name
        self.n = y.shape[0]

    def update_temperature_schedule(self):
        # Get temperature schedule function
        temperature_schedule_fn = tp.map_name_to_temperature_schedule(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_schedule'])
        # Get number of steps and power
        nsteps = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])])
        power = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_power'])])
        # Update class attribute
        self.temperature_schedule = temperature_schedule_fn(nsteps,power)

    def reflect_proposal(self,pnew,mcmc_type):
        # Find lower and upper bounds
        lower_bound = list(self.inference_metadata['inference'][mcmc_type]['transition_kernel']['lower_bound'])
        upper_bound = list(self.inference_metadata['inference'][mcmc_type]['transition_kernel']['upper_bound'])
        # Constrain proposal
        for i in range(len(pnew)):
            if pnew[i] <= lower_bound[i]:
                pnew[i] = 2*lower_bound[i]-pnew[i]
            if pnew[i] >= upper_bound[i]:
                pnew[i] = 2*upper_bound[i]-pnew[i]
        return pnew

    def reject_proposal(self,pnew,mcmc_type):
        # Find lower and upper bounds
        lower_bound = list(self.inference_metadata['inference'][mcmc_type]['transition_kernel']['lower_bound'])
        upper_bound = list(self.inference_metadata['inference'][mcmc_type]['transition_kernel']['upper_bound'])
        # Constrain proposal
        for i in range(len(pnew)):
            if pnew[i] <= lower_bound[i] or pnew[i] >= upper_bound[i]:
                return True
        return pnew


    def propose_new_sample(self,p):
        utils.validate_attribute_existence(self,['transition_kernel'])
        return self.__transition_kernel(p)

    def propose_new_sample_thermodynamic_integration(self,p):
        utils.validate_attribute_existence(self,['thermodynamic_integration_transition_kernel'])
        return self.__thermodynamic_integration_transition_kernel(p)

    def evaluate_log_target(self,p):
        utils.validate_attribute_existence(self,['log_target'])
        return self.__log_target(p)

    def evaluate_log_target_thermodynamic_integration(self,p):
        utils.validate_attribute_existence(self,['log_target_thermodynamic_integration'])
        return self.__log_target_thermodynamic_integration(p)

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

    def evaluate_log_kernel(self,p):
        utils.validate_attribute_existence(self,['log_kernel'])
        return self.__log_kernel(p)

    def evaluate_thermodynamic_integration_log_posterior(self,p,t):
        return self.evaluate_log_joint_prior(p[t,:]) + self.temperature_schedule[t]*self.evaluate_log_likelihood(p[t,:])

    def log_acceptance_ratio(pnew,pprev):
        log_pnew = self.evaluate_log_target(pnew) + self.evaluate_log_kernel(pnew,pprev)
        log_pold = self.evaluate_log_target(pprev) + self.evaluate_log_kernel(pprev,pnew)
        log_acc = log_pnew - log_pold
        # If exp floating point exceeded
        if log_acc >= 709: return 0, log_pnew, log_pold
        else: return log_acc, log_pnew, log_pold


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
            proposed samples from target distribution
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
        utils.validate_attribute_existence(self,['evaluate_log_target'])

        # Make sure you have stored necessary attributes
        utils.validate_parameter_existence(['N'],self.inference_metadata['inference']['vanilla_mcmc'])
        utils.validate_parameter_existence(['K_diagonal','beta_step'],self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel'])

        p0 = None
        # Read p0 or randomly initialise it from prior
        if utils.has_parameters(['p0'],self.inference_metadata['inference']['vanilla_mcmc']) and self.inference_metadata['inference']['vanilla_mcmc']['param_initialisation'] == 'metadata':
            p0 = np.array(self.inference_metadata['inference']['vanilla_mcmc']['p0'])
        elif utils.has_parameters(['p0'],self.inference_metadata['inference']['vanilla_mcmc']) and self.inference_metadata['inference']['vanilla_mcmc']['param_initialisation'] == 'mle':
            # Use MLE estimate
            p0 = np.array(self.mle_params)
            # Update metadata
            utils.update(self.__inference_metadata['inference']['thermodynamic_integration_mcmc'],{'p0':list(p0)})
        else:
            # Sample from prior distributions
            p0 = self.sample_from_univariate_priors(fundamental_diagram,1)
            # Update metadata
            utils.update(self.__inference_metadata['inference']['vanilla_mcmc'],{'p0':p0})

        # Initialise output variables
        theta = []
        theta_proposed = []
        acc = 0

        # Store necessary parameters
        p_prev = p0
        # Store number of iterations
        N = max(int(self.inference_metadata['inference']['vanilla_mcmc']['N']),1)
        burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['N'])

        if prints:
            print('p0',p_prev)
            print(f'Running MCMC with {N} iterations')

        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Propose new sample
            p_new = self.propose_new_sample(p_prev)

            # If rejection scheme for constrained theta applies
            if strtobool(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['constrained']) and self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['action'] == 'reflect':
                # If theta is NOT within lower and upper bounds reject sample
                p_new = self.reflect_proposal(p_new,'vanilla_mcmc')

            # Calculate acceptance probability
            if strtobool(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['constrained']) and self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['action'] == 'reject':
                # If theta is NOT within lower and upper bounds reject sample
                if self.reject_proposal(p_new,'vanilla_mcmc'): log_acc_ratio = -1e9
                lt_new = self.evaluate_log_target(p_new)
                lt_prev = self.evaluate_log_target(p_prev)
            else:
                log_acc_ratio,lt_new,lt_prev = self.log_acceptance_ratio(p_new,p_prev)

            # Compute acceptance probability
            acc_ratio = min(1,np.exp(log_acc_ratio))

            # Sample from Uniform(0,1)
            u = np.random.random()

            # Printing proposals every 0.1*Nth iteration
            if prints and (i in [int(j/10*N) for j in range(1,11)]):
                print('p_prev',p_prev,'lt_prev',lt_prev)
                print('p_new',p_new,'lt_new',lt_new)

            # Accept/Reject
            # Compare log_alpha and log_u to accept/reject sample
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

        # Update class attributes
        self.theta = np.array(theta).reshape((N,len(p0)))
        self.theta_proposed = np.array(theta_proposed).reshape((N,len(p0)))

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc": {"acceptance_rate":int(100*(acc / N))}})

        result_summary = {"vanilla_mcmc":{}}
        for i in range(self.theta.shape[1]):
            # Parameter name
            param_name = str(self.parameter_names[i]).replace("$","").replace("\\","")
            result_summary['vanilla_mcmc'][param_name] = {"mean":np.mean(self.theta[burnin:,i]),"std":np.std(self.theta[burnin:,i])}
        # Update metadata on results
        utils.update(self.__inference_metadata['results'],result_summary)

        if prints: print(f'Acceptance rate {int(100*(acc / N))}%')

        return np.array(theta).reshape((N,len(p0))), np.array(theta_proposed).reshape((N,len(p0))), int(100*(acc / N))

    def thermodynamic_integration_mcmc(self,fundamental_diagram,prints:bool=False,seed:int=None):
        """Thermodynamic integration MCMC method for sampling from pdf defined by log_function

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
        random.seed(seed)

        # Make sure you have stored necessary attributes
        utils.validate_attribute_existence(self,['evaluate_log_target','temperature_schedule'])

        # Make sure you have stored necessary attributes
        utils.validate_parameter_existence(['N'],self.inference_metadata['inference']['thermodynamic_integration_mcmc'])
        # utils.validate_parameter_existence(['K_diagonal','beta_step'],self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel'])

        # Store length of temperature schedule
        t_len = self.temperature_schedule.shape[0]

        p0 = None
        # Read p0 or randomly initialise it from prior
        if utils.has_parameters(['p0'],self.inference_metadata['inference']['thermodynamic_integration_mcmc']) and self.inference_metadata['inference']['thermodynamic_integration_mcmc']['param_initialisation'] == 'metadata':
            # Use p0 from inference metadata
            p0 = np.array(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['p0'])
            # Repeat n times
            p0 = np.repeat([p0],t_len,axis=0)
        elif utils.has_parameters(['p0'],self.inference_metadata['inference']['thermodynamic_integration_mcmc']) and self.inference_metadata['inference']['thermodynamic_integration_mcmc']['param_initialisation'] == 'mle':
            # Use MLE estimate
            p0 = np.array(self.mle_params)
            # Update metadata
            utils.update(self.__inference_metadata['inference']['thermodynamic_integration_mcmc'],{'p0':list(p0)})
            # Repeat n times
            p0 = np.repeat([p0],t_len,axis=0)
        else:
            # Sample from prior distributions
            p0 = self.sample_from_univariate_priors(fundamental_diagram,t_len).reshape((t_len,self.num_learning_parameters))
            # Update metadata
            utils.update(self.__inference_metadata['inference']['thermodynamic_integration_mcmc'],{'p0':list(p0)})

        # Store necessary parameters
        p_prev = p0
        # Store number of iterations
        N = np.max([int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),1])

        # Initialise output variables
        theta = np.zeros((N,t_len,p_prev.shape[1]))
        acc,prop = np.zeros(t_len),np.zeros(t_len)

        if prints:
            print('p0',p_prev)
            print(f'Running MCMC with {N} iterations')

        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Select random temperature from schedule
            random_t_index = np.random.randint(0,t_len)

            # Copy previously accepted sample
            p_new = copy.deepcopy(p_prev)

            # Copy previously accepted sample
            p_new[random_t_index,:] = self.propose_new_sample_thermodynamic_integration(p_prev[random_t_index,:])

            # Evaluate log function for current sample
            lt_prev = self.evaluate_thermodynamic_integration_log_target(p_prev,random_t_index)

            # Evaluate log function for proposed sample
            lt_new = self.evaluate_thermodynamic_integration_log_target(p_new,random_t_index)

            # Printing proposals every 0.1*Nth iteration
            if prints and (i in [int(j/10*N) for j in range(1,11)]):
                print('p_prev',p_prev[random_t_index,:],'lt_prev',lt_prev)
                print('p_new',p_new[random_t_index,:],'lt_new',lt_new)

            # Calculate acceptance probability
            # If rejection scheme for constrained theta applies
            if self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['action'] == 'reject':
                # If theta is NOT within lower and upper bounds reject sample
                if self.reject_proposal(p_new,'thermodynamic_integration_mcmc'): log_acc = -1e9
                # Calculate acceptance probability
                else: log_acc = lt_new - lt_prev
            else:
                log_acc = lt_new - lt_prev
            # Sample from Uniform(0,1)
            log_u = np.log(np.random.random())

            # Increment proposal counter
            prop[random_t_index] += 1

            # Accept/Reject
            # Compare log_alpha and log_u to accept/reject sample
            if min(np.exp(log_acc),1) >= np.exp(log_u):
                # Increment accepted sample count
                acc[random_t_index] += 1
                # Append to accepted and proposed sample arrays
                theta[i,:,:] = p_new
                # Update last accepted sample
                p_prev = p_new
            else:
                # Append to accepted and proposed sample arrays
                theta[i,:,:] = p_prev

            if prints and (i in [int(j/10*N) for j in range(1,11)]):
                print('Rejected...')
                print(f'i = {random_t_index}, t_i = {self.temperature_schedule[random_t_index]}, p_new = {p_new[random_t_index,:]}')
                print(f'Total acceptance rate {int(100*np.sum(acc) / np.sum(prop))}%')
                print(f'Temperature acceptance rate {int(100*acc[random_t_index] / prop[random_t_index])}%')


        # Update class attributes
        self.thermodynamic_integration_theta = np.array(theta)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc": {"acceptance_rate":[int(100*a) for a in acc/prop],"acceptances":list(acc)},"proposals":list(prop)})

        result_summary = {"thermodynamic_integration_mcmc":{}}
        for i in range(self.thermodynamic_integration_theta.shape[2]):
            # Parameter name
            param_name = str(self.parameter_names[i]).replace("$","") .replace("\\","")
            result_summary['thermodynamic_integration_mcmc'][param_name] = {"mean":list(np.mean(self.thermodynamic_integration_theta[:,:,i],axis=0)),"std":list(np.std(self.thermodynamic_integration_theta[:,:,i],axis=0))}
        # Update metadata on results
        utils.update(self.__inference_metadata['results'],result_summary)

        if prints:
            print(f'Total acceptance rate {int(100*(np.sum(acc) / np.sum(prop)))}%')
            print(f"Temperature acceptance rate {[(str(int(100*a))+'%') for a in acc/prop]}")
            print(f"Choice probabilities {[(str(int(100*pr/N))+'%') for pr in prop]}")

        return np.array(theta), acc, prop

    def compute_maximum_likelihood_estimate(self,fundamental_diagram,**kwargs):

        warnings.simplefilter("ignore")

        # Make sure you have imported metadata
        utils.validate_attribute_existence(self,['inference_metadata','evaluate_log_target'])
        utils.validate_attribute_existence(fundamental_diagram,['num_learning_parameters','true_parameters'])

        # Fix random seed
        if fundamental_diagram.simulation_metadata['seed'] == '' or fundamental_diagram.simulation_metadata['seed'].lower() == 'none':
            np.random.seed(None)
        else:
            np.random.seed(int(fundamental_diagram.simulation_metadata['seed']))


        # Get p0 from metadata
        p0 = list(self.inference_metadata['inference']['mle']['p0'])

        # If no p0 is provided set p0 to true parameter values
        if len(p0) < 1:
            if (not self.inference_metadata['simulation_flag']):
                raise ValueError('True parameters cannot be found for MLE estimator initialisation.')
            else:
                p0 = self.true_parameters

        # Define negative of log likelihood
        def negative_log_likelihood(p):
            return (-1)*self.evaluate_log_likelihood(p)

        # Optimise parameters for negative_log_target
        mle = fmin(negative_log_likelihood, p0, disp = False)[0:self.num_learning_parameters]

        # Get parameters
        self.mle_params = mle
        # Evaluate log target
        mle_log_likelihood = self.evaluate_log_likelihood(self.mle_params)
        true_log_likelihood = self.evaluate_log_likelihood(self.true_parameters)
        true_log_target = self.evaluate_log_target(self.true_parameters)
        true_log_prior = self.evaluate_log_joint_prior(self.true_parameters)

        # Update class variables
        utils.update(self.__inference_metadata['results'],{"mle":{"params":self.mle_params,"mle_log_likelihood":mle_log_likelihood,"true_log_likelihood":true_log_likelihood}})

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                print(f'MLE parameters: {self.mle_params}')
                print(f'MLE log likelihood: {mle_log_likelihood}')
                print(f'True log prior: {true_log_prior}')
                print(f'True log likelihood: {true_log_likelihood}')
                print(f'True log target: {true_log_target}')

    def compute_log_posterior_harmonic_mean_estimator(self,**kwargs):

        # Make sure you have stored necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        if 'prints' in kwargs:
            if kwargs.get('prints'): print('Computing marginal likelihood estimate based on vanilla MCMC')

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = np.max([int(self.inference_metadata['plot']['vanilla_mcmc']['burnin']),0])

        # Get number of MCMC iterations
        N = self.theta.shape[0]

        # for i in range(burnin,N):
        #     term = np.exp(self.evaluate_log_likelihood(self.theta[i,:]))**(-1)
        #     print(self.log_y)
        #     print('param posterior',self.theta[i,:])
        #     print('term',term)
        #     sys.exit(1)

        # Compute log marginal likelihood
        ml = N * ( np.sum([np.exp(self.evaluate_log_likelihood(self.theta[i,:]))**(-1) for i in range(burnin,N)]) )**(-1)
        lml = np.log(ml)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"log_marginal_likelihood":lml}})

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                # Print log marginal likelihood
                print(f'Log marginal likelihood = {lml}')
                # Print time execution
                toc = time.perf_counter()
                print(f"Computed posterior harmonic mean estimator in {toc - tic:0.4f} seconds")

        return lml

    def compute_thermodynamic_integration_log_marginal_likelihood_estimator(self,**kwargs):

        # Make sure you have stored necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        if 'prints' in kwargs:
            if kwargs.get('prints'): print('Computing marginal likelihood estimate based on thermodynamic integration MCMC')

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = np.max([int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['burnin']),0])

        # Get number of MCMC iterations
        N = self.thermodynamic_integration_theta.shape[0]

        # Store length of temperature schedule
        t_len = self.temperature_schedule.shape[0]

        # lml = 0
        # for ti in range(1,t_len):
        #     for j in range(burnin,N):
        #         term1 = self.evaluate_log_likelihood(self.thermodynamic_integration_theta[j,ti,:])
        #         term2 = self.evaluate_log_likelihood(self.thermodynamic_integration_theta[j,ti-1,:])
        #         print('j =',j)
        #         print('thermodynamic_integration_theta[j,ti,:]',self.thermodynamic_integration_theta[j,ti,:])
        #         print('deltat',(self.temperature_schedule[ti] - self.temperature_schedule[ti-1]))
        #         print('term1',term1)
        #         print('term2',term2)
        #         sys.exit(1)
        #         term = (self.temperature_schedule[ti] - self.temperature_schedule[ti-1])*(term1-term2)
        #         lml += term

        # Initiliase lml
        lml = np.sum([(self.temperature_schedule[ti] - self.temperature_schedule[ti-1])*(self.evaluate_log_likelihood(self.thermodynamic_integration_theta[j,ti,:]) + self.evaluate_log_likelihood(self.thermodynamic_integration_theta[j,ti-1,:])) for ti in range(1,t_len) for j in range(burnin,N)])
        lml /= 2*(N-burnin)
        if 'prints' in kwargs:
            if kwargs.get('prints'):
                # Print log marginal likelihood
                print(f'Log marginal likelihood = {lml}')
                # Print time execution
                toc = time.perf_counter()
                print(f"Computed thermodynamic integration marginal likelihood estimator in {toc - tic:0.4f} seconds")

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"log_marginal_likelihood":float(lml)}})

        return lml


    def compute_gelman_rubin_statistic(self,theta,mcmc_type:int='thermodynamic_integration_mcmc',**kwargs):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot'][mcmc_type]['burnin'])

        # Make sure you have the necessary parameters
        utils.validate_parameter_existence(['r_critical'],self.inference_metadata['inference'][mcmc_type])

        # Get R statistic critical value
        r_critical = float(self.inference_metadata['inference'][mcmc_type]['r_critical'])

        # Get number of chain iterations and number of chains
        n,m = theta.shape

        # If burnin exists error there is a problem
        if burnin >= n: raise ValueError(f'Burnin {burnin} cannot be >= to MCMC iterations {n}')

        # Compute posterior mean for each parameter dimension
        posterior_parameter_means = np.array([np.mean(theta[burnin:,j]) for j in range(m)])
        # Compute B
        B = (n-burnin)/(m-1) * np.sum([(posterior_parameter_means[j] - np.mean(theta[burnin:,:],axis=(0,1)))**2 for j in range(m)])
        # Compute W
        W = (1./m) * np.sum([(1./((n-burnin)-1)* np.sum([(theta[i,j]-posterior_parameter_means[j])**2 for i in range(burnin,n)])) for j in range(m)])
        # Compute parameter marginal posterior variance
        posterior_marginal_var = (((n-burnin)-1)/(n-burnin))*W + B/(n-burnin)
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
                print(f"Computed Gelman & Rubin estimator in {toc - tic:0.4f} seconds")
        return r_stat


    def compute_gelman_rubin_statistic_for_vanilla_mcmc(self,**kwargs):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Make sure you have the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Compute Gelman and Rubin statistic for vanilla MCMC chain
        r_stat = self.compute_gelman_rubin_statistic(self.theta)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"r_stat":r_stat}})
        return r_stat

    def compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(self,**kwargs):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Make sure you have the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Time execution
        tic = time.perf_counter()


        # Initialise array of r stats
        r_stats = []
        for ti in range(len(self.temperature_schedule)):
            # Compute Gelman and Rubin statistic for vanilla MCMC chain
            r_stat = self.compute_gelman_rubin_statistic(self.thermodynamic_integration_theta[:,ti,:])
            # Append to array
            r_stats.append(r_stat)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"r_stat":r_stats}})

        if 'prints' in kwargs:
            if kwargs.get('prints'):
                print('R statistics:',r_stats)
                # Print time execution
                toc = time.perf_counter()
                print(f"Computed Gelman & Rubin estimator in {toc - tic:0.4f} seconds")

        return r_stats




    """---------------------------------------------------------------------------Evaluate and generate posterior data/plots-----------------------------------------------------------------------------"""

    def evaluate_posterior_predictive_moments(self,*args,seed:int=None,**kwargs):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['vanilla_mcmc']['burnin'])

        # Set posterior predictive range to covariate range
        x = self.x
        # Set posterior predictive x based on args
        if len(args) == 1 and hasattr(args[0],'__len__'):
            x = args[0]

        # Fix random seed
        np.random.seed(seed)

        # Sample from predictive likelihood
        pp = np.array([self.evaluate_predictive_likelihood(self.theta[j,:],x) for j in range(burnin,self.theta.shape[0])])
        # Compute posterior predictive mean
        pp_mean = np.mean(pp,axis=0)
        # Compute posterior predictive standard deviation
        pp_var = np.mean(pp**2,axis=0) - pp_mean**2

        # Update class variables
        self.posterior_predictive_mean = pp_mean
        self.posterior_predictive_std = np.sqrt(pp_var)
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
        num_params = self.num_learning_parameters
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

        # Vectorize evaluate_log_target
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
        # fig, axs = plt.subplots(figsize=(10,10*fundamental_diagram.num_learning_parameters),nrows=fundamental_diagram.num_learning_parameters,ncols=1)

        # Get prior distribution parameters
        prior_params = list(self.inference_metadata['inference']['priors'].values())
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter number
        for i in range(0,self.num_learning_parameters):

            fig = plt.figure(figsize=(10,8))

            # Define x range
            xrange = np.linspace(0,3,1000)
            # Store prior hyperparameter kwargs from metadata
            hyperparams = {}
            prior_key = list(self.inference_metadata['inference']['priors'].keys())[i]
            for k, v in self.inference_metadata['inference']['priors'][prior_key].items():
                if k != "distribution":
                    hyperparams[k] = float(v)

            yrange = self.log_univariate_priors[i](xrange,**hyperparams)[0]
            prior_mean = np.round(self.log_univariate_priors[i](xrange,**hyperparams)[1],5)
            prior_std = np.round(self.log_univariate_priors[i](xrange,**hyperparams)[2],5)

            # Store distributio and parameter names
            distribution_name = prior_params[i]['distribution'].capitalize()
            parameter_name = self.parameter_names[i]

            # Plot pdf
            plt.plot(xrange,yrange,color='blue',label='pdf')
            # Plot prior mean
            plt.vlines(prior_mean,ymin=-1,ymax=np.max(yrange[np.isfinite(yrange)]),color='red',label=f'mean = {prior_mean}')
            # Plot prior mean +/- prior std
            plt.hlines(np.max(yrange)/2,xmin=(prior_mean-prior_std),xmax=(prior_mean+prior_std),color='green',label=f'mean +/- std, std = {prior_std}')

            # Print hyperparameters
            if prints: print(f'Prior hypeparameters for {self.parameter_names[i]}:',', '.join(['{}={!r}'.format(k, v) for k, v in hyperparams.items()]))

            # Plot true parameter if it exists
            if hasattr(self,'true_parameters'):
                plt.vlines(self.true_parameters[i],ymin=0,ymax=np.max(yrange[np.isfinite(yrange)]),color='black',label='true',linestyle='dashed')
            # Change x limit
            print('X limit needs fixing')
            # plt.xlim(0,self.log_univariate_priors[i].interval(.999,**hyperparams)[1])
            # Change y limit
            plt.ylim(0,np.max(yrange[np.isfinite(yrange)])*100/99)
            # Set title
            if show_titles: plt.title(f"{distribution_name} prior for {parameter_name} parameter")
            # Plot legend
            plt.legend()

            # Plot figure
            if show_plot: plt.show()
            # Append to figures
            figs.append({"parameters":[self.parameter_names[i]],"figure":fig})
            # Close plot
            plt.close(fig)


        return figs


    def generate_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['vanilla_mcmc']['burnin'])
        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):

            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Add samples plot
            plt.plot(range(burnin,self.theta.shape[0]),self.theta[burnin:,p],color='blue',label='Samples',zorder=1)

            # Plot true parameters if they exist
            if hasattr(self,'true_parameters'):
                plt.hlines(self.true_parameters[p],xmin=burnin,xmax=(self.theta.shape[0]),color='black',label='True',zorder=5)

            print(self.parameter_names[p])
            print('mean',np.mean(self.theta[burnin:,p],axis=0))

            # Plot inferred mean
            plt.hlines(np.mean(self.theta[burnin:,p],axis=0),xmin=burnin,xmax=(self.theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=4)

            # Add labels
            plt.xlabel('MCMC Iterations')
            plt.ylabel(f'MCMC Samples')
            if show_titles: plt.title(f'Mixing for {self.parameter_names[p]} with burnin = {burnin}')

            # Add legend
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[self.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_thermodynamic_integration_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):
        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.thermodynamic_integration_theta.shape[2]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['burnin'])
        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter indices
        for ti in range(self.thermodynamic_integration_theta.shape[1]):

            # Loop through parameter indices
            for p in range(self.thermodynamic_integration_theta.shape[2]):

                # Generate figure
                fig = plt.figure(figsize=(10,8))

                # Add samples plot
                plt.plot(range(burnin,self.thermodynamic_integration_theta.shape[0]),self.thermodynamic_integration_theta[burnin:,ti,p],color='blue',label='Samples')

                # Plot true parameters if they exist
                if hasattr(self,'true_parameters'):
                    plt.hlines(self.true_parameters[p],xmin=burnin,xmax=(self.thermodynamic_integration_theta.shape[0]),label='True',color='black')

                # Plot inferred mean
                plt.hlines(np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0),xmin=burnin,xmax=(self.thermodynamic_integration_theta.shape[0]),color='red',label=f'Posterior $\mu$',linestyle='dashed')

                # Add labels
                plt.xlabel('MCMC Iterations')
                plt.ylabel(f'MCMC Samples')
                if show_titles: plt.title(f'Mixing for {self.parameter_names[p]}, t = {np.round(self.temperature_schedule[ti],5)} with burnin = {burnin}')

                # Add legend
                plt.legend()

                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[self.parameter_names[p]+("_temperature_")+str(ti)],"figure":fig})
                # Close current plot
                plt.close(fig)

        return figs


    def generate_mcmc_acf_plots(self,fundamental_diagram,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['plot']['vanilla_mcmc']['burnin'])
        lags = np.min([int(self.inference_metadata['plot']['vanilla_mcmc']['acf_lags']),(self.theta[burnin:,:].shape[0]-1)])
        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):
            # Generate figure
            fig,ax = plt.subplots(1,figsize=(10,8))

            # Add ACF plot
            if show_titles: sm.graphics.tsa.plot_acf(self.theta[burnin:,p], ax=ax, lags=lags, title=f'ACF plot for {self.parameter_names[p]} with burnin = {burnin}')
            else: sm.graphics.tsa.plot_acf(self.theta[burnin:,p], ax=ax, lags=lags, title="")

            # Add labels
            ax.set_ylabel(f'{self.parameter_names[p]}')
            ax.set_xlabel('Lags')

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[self.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_mcmc_parameter_posterior_plots(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.theta.shape[1]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and histogram bins from plot metadata
        burnin = int(self.inference_metadata['plot']['vanilla_mcmc']['burnin'])
        bins = np.max([int(self.inference_metadata['plot']['vanilla_mcmc']['hist_bins']),10])
        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter indices
        for p in range(self.theta.shape[1]):
            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Plot parameter posterior
            freq,_,_ = plt.hist(self.theta[burnin:,p],bins=bins)

            # Compute posterior mean
            sample_mean = np.mean(self.theta[burnin:,p])
            # sigma2 = self.true_parameters[-1]
            # var = np.log(1+sigma2)

            posterior_mean = sample_mean #*np.exp(var/2 - 3*var/(2*len(self.theta[burnin:,p])))

            # print('posterior_mean',posterior_mean)
            # print('sample_mean',sample_mean)

            # Compute posterior std
            posterior_std = np.std(self.theta[burnin:,p])

            # Add labels
            if show_titles: plt.title(f'Parameter posterior for {self.parameter_names[p]} with burnin = {burnin}')
            plt.vlines(posterior_mean,0,np.max(freq),color='red',label=r'$\mu$', linewidth=2)
            plt.vlines(posterior_mean-num_stds*posterior_std,0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
            plt.vlines(posterior_mean+num_stds*posterior_std,0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
            # Plot true parameters if they exist
            if hasattr(self,'true_parameters'):
                plt.vlines(self.true_parameters[p],0,np.max(freq),label='True',color='black',linewidth=2)
            plt.xlabel(f'{self.parameter_names[p]}')
            plt.ylabel('Sample frequency')
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[self.parameter_names[p]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs


    def generate_thermodynamic_integration_mcmc_parameter_posterior_plots(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):
        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.thermodynamic_integration_theta.shape[2]:
            raise ValueError(f'Posterior has {self.theta.shape[1]} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and histogram bins from plot metadata
        burnin = int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['burnin'])
        bins = np.max([int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['hist_bins']),10])
        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        figs = []
        # Loop through parameter indices
        for ti in range(self.thermodynamic_integration_theta.shape[1]):

            # Loop through parameter indices
            for p in range(self.thermodynamic_integration_theta.shape[2]):

                # Generate figure
                fig = plt.figure(figsize=(10,8))

                # Plot parameter posterior
                freq,_,_ = plt.hist(self.thermodynamic_integration_theta[burnin:,ti,p],bins=bins)

                # Add labels
                if show_titles: plt.title(f'Parameter posterior for {self.parameter_names[p]} t = {self.temperature_schedule[ti]} with burnin = {burnin}')
                plt.vlines(np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0),0,np.max(freq),color='red',label=r'$\mu$', linewidth=2)
                plt.vlines((np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)-num_stds*np.std(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)),0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
                plt.vlines((np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)+num_stds*np.std(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)),0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
                # Plot true parameters if they exist
                if hasattr(self,'true_parameters'):
                    plt.vlines(self.true_parameters[p],0,np.max(freq),label='True',color='black',linewidth=2)
                plt.xlabel(f'{self.parameter_names[p]}')
                plt.ylabel('Sample frequency')
                plt.legend()


                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[self.parameter_names[p]+("_temperature_")+str(ti)],"figure":fig})
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
        parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.num_learning_parameters), 2))
        parameter_names = list(itertools.combinations(self.parameter_names, 2))

        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

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
            burnin = int(self.inference_metadata['plot']['vanilla_mcmc']['burnin'])

            # Get parameters to plot
            theta_subset = self.theta[burnin:,list(index)]
            theta_proposed_subset = self.theta_proposed[burnin:,list(index)]

            # Add samples plot
            plt.scatter(theta_subset[:,index[0]],theta_subset[:,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=5)
            plt.scatter(theta_proposed_subset[:,index[0]],theta_proposed_subset[:,index[1]],color='purple',label='Proposed',marker='x',s=50,zorder=4)

            # Get log unnormalised posterior plot
            if include_posterior:
                utils.validate_attribute_existence(self,['log_unnormalised_posterior'])
                # Set Q_hat to log posterior
                Q_hat = self.log_unnormalised_posterior
                # Sum up dimension not plotted if there log posterior is > 2dimensional
                if len(Q_hat.shape) > 2:
                    Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.num_learning_parameters)) - set(index))[0])

                # Try to load plot parameters
                levels = None
                # Check if all plot parameters are not empty
                if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
                    # Get number of colors in contour
                    num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                    # Update levels
                    levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
                else:
                    vmin = np.min(Q_hat)
                    if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                        vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
                    vmax = np.max(Q_hat)
                    if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                        vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
                    num_colors = int(np.sqrt(np.prod(Q_hat.shape)))
                    if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                        # Get number of colors in contour
                        num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                    # Get levels
                    if vmin >= vmax: print('Wrong order of vmin, vmax in unnormalised posterior plot'); levels = np.linspace(vmax,vmin,num_colors)
                    else: levels = np.linspace(vmin,vmax,num_colors)

                # Plot countour surface
                im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels,zorder=1)
                # Plot MAP estimate
                plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=6)
                # Change limits
                plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
                plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
                # Plot colorbar
                plt.colorbar(im)
            else:
                # Get limits from plotting metadata
                plot_limits = self.inference_metadata['plot']['vanilla_mcmc']

                plt.xlim([float(plot_limits['xmin']),float(plot_limits['xmax'])])
                plt.ylim([float(plot_limits['ymin']),float(plot_limits['ymax'])])

            # Plot true parameters if they exist
            if hasattr(self,'true_parameters'):
                plt.scatter(self.true_parameters[index[0]],self.true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=7)

            # Add labels
            plt.xlabel(f'{parameter_names[i][index[0]]}')
            plt.ylabel(f'{parameter_names[i][index[1]]}')
            # Add title
            if show_titles: plt.title(f'{parameter_names[i][index[0]]},{parameter_names[i][index[1]]} space exploration with burnin = {burnin}')
            # Add legend
            plt.legend()

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[parameter_names[i][index[0]],parameter_names[i][index[1]]],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_thermodynamic_integration_mcmc_space_exploration_plots(self,fundamental_diagram,include_posterior:bool=False,show_plot:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Get starting time
        start = time.time()

        # Get number of plots
        num_plots = int(comb(self.thermodynamic_integration_theta.shape[2],2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.num_learning_parameters), 2))
        parameter_names = list(itertools.combinations(self.parameter_names, 2))

        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])
        # Get burnin
        burnin = int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['burnin'])

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

            for tj in range(len(self.temperature_schedule)):

                # Create figure
                fig = plt.figure(figsize=(10,8))

                # Get parameters to plot
                theta_subset = self.thermodynamic_integration_theta[burnin:,tj,list(index)]

                # Add samples plot
                plt.scatter(theta_subset[:,index[0]],theta_subset[:,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=2)

                # Get log unnormalised posterior plot
                if include_posterior:
                    utils.validate_attribute_existence(self,['log_unnormalised_posterior'])
                    # Set Q_hat to log posterior
                    Q_hat = self.log_unnormalised_posterior
                    # Sum up dimension not plotted if there log posterior is > 2dimensional
                    if len(Q_hat.shape) > 2:
                        Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.num_learning_parameters)) - set(index))[0])

                    # Try to load plot parameters
                    levels = None
                    # Check if all plot parameters are not empty
                    if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
                        # Get number of colors in contour
                        num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                        # Update levels
                        levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
                    else:
                        vmin = np.min(Q_hat)
                        if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                            vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
                        vmax = np.max(Q_hat)
                        if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                            vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
                        num_colors = int(np.sqrt(np.prod(Q_hat.shape)))
                        if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                            # Get number of colors in contour
                            num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                        # Get levels
                        if vmin >= vmax: print('Wrong order of vmin, vmax in unnormalised posterior plot'); levels = np.linspace(vmax,vmin,num_colors)
                        else: levels = np.linspace(vmin,vmax,num_colors)

                    # Plot countour surface
                    im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels,zorder=1)
                    # Plot MAP estimate
                    plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=3)
                    # Change limits
                    xmin = np.min([np.min(self.parameter_mesh[index[0]]),np.min(theta_subset[:,index[0]])])
                    xmax = np.max([np.max(self.parameter_mesh[index[0]]),np.max(theta_subset[:,index[0]])])
                    ymin = np.min([np.min(self.parameter_mesh[index[1]]),np.min(theta_subset[:,index[1]])])
                    ymax = np.max([np.max(self.parameter_mesh[index[1]]),np.max(theta_subset[:,index[1]])])

                    plt.ylim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
                    plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
                    # Plot colorbar
                    plt.colorbar(im)
                else:
                    # Get limits from plotting metadata
                    plot_limits = self.inference_metadata['plot']['thermodynamic_integration_mcmc']

                    xmin = np.min([float(plot_limits['xmin']),np.min(theta_subset[:,index[0]])])
                    xmax = np.max([float(plot_limits['xmax']),np.max(theta_subset[:,index[0]])])
                    ymin = np.min([float(plot_limits['ymin']),np.min(theta_subset[:,index[1]])])
                    ymax = np.max([float(plot_limits['ymax']),np.max(theta_subset[:,index[1]])])

                # Change limits
                plt.xlim([xmin,xmax])
                plt.ylim([ymin,ymax])

                # Plot true parameters if they exist
                if hasattr(self,'true_parameters'):
                    plt.scatter(self.true_parameters[index[0]],self.true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=4)

                # Add labels
                plt.xlabel(f'{parameter_names[i][index[0]]}')
                plt.ylabel(f'{parameter_names[i][index[1]]}')
                # Add title
                if show_titles: plt.title(f'{parameter_names[i][index[0]]},{parameter_names[i][index[1]]} space exploration with t = {np.round(self.temperature_schedule[tj],5)} burnin = {burnin}')
                # Add legend
                plt.legend()

                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[parameter_names[i][index[0]],parameter_names[i][index[1]],f'temperature_{tj}'],"figure":fig})
                # Close current plot
                plt.close(fig)

        return figs


    def generate_log_unnormalised_posteriors_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get starting time
        start = time.time()

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['log_unnormalised_posterior'])

        # Get number of plots
        num_plots = int(comb(len(self.log_unnormalised_posterior.shape),2))

        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,self.num_learning_parameters), 2))
        parameter_names = list(itertools.combinations(self.parameter_names, 2))

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
                Q_hat = np.sum(Q_hat,axis=list(set(range(0,self.num_learning_parameters)) - set(index))[0])

            # Try to load plot parameters
            levels = None
            # Check if all plot parameters are not empty
            if all(bool(x) for x in self.inference_metadata['plot']['true_posterior'].values()):
                # Get number of colors in contour
                num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                # Update levels
                levels = np.linspace(float(self.inference_metadata['plot']['true_posterior']['vmin']),float(self.inference_metadata['plot']['true_posterior']['vmax']),num_colors)
            else:
                vmin = np.min(Q_hat)
                if bool(self.inference_metadata['plot']['true_posterior']['vmin']):
                    vmin = np.max([float(self.inference_metadata['plot']['true_posterior']['vmin']),np.min(Q_hat)])
                vmax = np.max(Q_hat)
                if bool(self.inference_metadata['plot']['true_posterior']['vmax']):
                    vmax = np.min([float(self.inference_metadata['plot']['true_posterior']['vmax']),np.max(Q_hat)])
                num_colors = int(np.sqrt(np.prod(Q_hat.shape)))
                if bool(self.inference_metadata['plot']['true_posterior']['num_colors']):
                    # Get number of colors in contour
                    num_colors = np.max([int(self.inference_metadata['plot']['true_posterior']['num_colors']),2])
                # Get levels
                if vmin >= vmax: print('Wrong order of vmin, vmax in unnormalised posterior plot'); levels = np.linspace(vmax,vmin,num_colors)
                else: levels = np.linspace(vmin,vmax,num_colors)

            # Create figure
            fig = plt.figure(figsize=(10,8))

            # Plot countour surface
            im = plt.contourf(self.parameter_mesh[index[0]], self.parameter_mesh[index[1]], Q_hat, levels=levels)

            plt.scatter(self.parameter_mesh[index[0]].flatten()[np.argmax(Q_hat)],self.parameter_mesh[index[1]].flatten()[np.argmax(Q_hat)],label='surface max',marker='x',s=200,color='blue',zorder=10)
            if hasattr(self,'true_parameters'):
                plt.scatter(self.true_parameters[index[0]],self.true_parameters[index[1]],label='True',marker='x',s=100,color='black',zorder=11)
            plt.xlim([np.min(self.parameter_mesh[index[0]]),np.max(self.parameter_mesh[index[0]])])
            plt.ylim([np.min(self.parameter_mesh[index[1]]),np.max(self.parameter_mesh[index[1]])])
            if show_titles: plt.title(f'Log unnormalised posterior for {",".join(parameter_names[i])}')
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

        print('Needs fixing')
        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        # Get starting time
        start = time.time()

        figs = []

        # Get flag for showing titles
        show_titles = strtobool(self.inference_metadata['plot']['show_titles'])

        # Create figure
        fig = plt.figure(figsize=(10,8))

        # Compute upper and lower bounds
        q_upper = self.posterior_predictive_mean + num_stds*self.posterior_predictive_std
        q_mean = self.posterior_predictive_mean
        q_lower= self.posterior_predictive_mean - num_stds*self.posterior_predictive_std

        plt.scatter(self.x,self.log_y,label='Observed data',color='blue',zorder=2,s=10)
        plt.plot(self.posterior_predictive_x,q_mean,color='red',label=r'$\mu$',zorder=1)
        if hasattr(fundamental_diagram,'true_parameters'):
            q_true = fundamental_diagram.simulate_with_x(self.true_parameters,self.posterior_predictive_x)
            plt.plot(fundamental_diagram.rho,q_true,color='black',label='true',zorder=2)
        plt.fill_between(self.posterior_predictive_x,q_upper,q_lower,alpha=0.5,color='red',label=f"$\mu$ +/- {num_stds}$\sigma$",zorder=3)
        if show_titles: plt.title(f"Posterior predictive for {self.inference_metadata['fundamental_diagram']} FD")
        plt.xlabel(f'{self.x_name}')
        plt.ylabel(f'{self.log_y_name}')
        plt.legend()

        # Show plot
        if show_plot: plt.show()
        # Append plot to list
        figs.append({"parameters":self.parameter_names,"figure":fig})
        # Close current plot
        plt.close(fig)

        return figs





    """ ---------------------------------------------------------------------------Import data-----------------------------------------------------------------------------"""


    def import_metadata(self,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Make sure file exists
        if not os.path.exits((inference_filename+'metadata.json')):
            raise FileNotFoundError(f"Metadata file {'metadata.json'} not found")

        #  Import metadata where acceptance is part of metadata
        with open((inference_filename+'metadata.json')) as json_file:
            self.inference_metadata = json.load(json_file)

        if 'prints' in kwargs:
            if kwargs.get('prints'):print('Imported MCMC samples')


    def import_mcmc_samples(self,fundamental_diagram,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        # Get burnins for Vanilla MCMC and Thermodynamic integration MCMC
        vanilla_burnin = np.max([int(self.inference_metadata['plot']['vanilla_mcmc']['burnin']),0])
        ti_burnin = np.max([int(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['burnin']),0])

        # Load theta from txt file
        file = (inference_filename+f'theta.txt')
        if os.path.exists(file):
            self.theta = np.loadtxt(file)

            # Update posterior mean and std in inference metadata results section
            result_summary = {"vanilla_mcmc":{}}
            for i in range(self.theta.shape[1]):
                # Parameter name
                param_name = str(self.parameter_names[i]).replace("$","") .replace("\\","")
                result_summary['vanilla_mcmc'][param_name] = {"mean":np.mean(self.theta[vanilla_burnin:,i]),"std":np.std(self.theta[vanilla_burnin:,i])}
            # Update metadata on results
            utils.update(self.__inference_metadata['results'],result_summary)


        # Load theta proposed from txt file
        file = (inference_filename+f'theta_proposed.txt')
        if os.path.exists(file):
            self.theta_proposed = np.loadtxt(file)


        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        # Load thermodynamic integration theta from txt file
        file = (inference_filename+f'thermodynamic_integration_theta.txt')
        if os.path.exists(file):
            self.thermodynamic_integration_theta = np.loadtxt(file,dtype='float64')
            # Reshape
            self.thermodynamic_integration_theta = self.thermodynamic_integration_theta.reshape((int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),len(self.temperature_schedule),self.num_learning_parameters))

            # Update posterior mean and std in inference metadata results section
            result_summary = {"thermodynamic_integration_mcmc":{}}
            for i in range(self.thermodynamic_integration_theta.shape[2]):
                # Parameter name
                param_name = str(self.parameter_names[i]).replace("$","") .replace("\\","")
                result_summary['thermodynamic_integration_mcmc'][param_name] = {"mean":list(np.mean(self.thermodynamic_integration_theta[ti_burnin:,:,i],axis=0)),"std":list(np.std(self.thermodynamic_integration_theta[ti_burnin:,:,i],axis=0))}
            # Update metadata on results
            utils.update(self.__inference_metadata['results'],result_summary)

        if 'prints' in kwargs:
            if kwargs.get('prints'):print('Imported MCMC samples')


    def import_log_unnormalised_posterior(self,parameter_pair:list,**kwargs):

        # Get rid of weird characters
        parameter_pair = [p.replace("$","").replace("\\","") for p in parameter_pair]

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Get parameter names
        param_names = "_".join([str(p).replace("$","").replace("\\","") for p in parameter_pair])
        # print('Importing unnormalised posterior')
        # Load from txt file
        try:
            file = inference_filename+f'log_unnormalised_posterior_{param_names}.txt'
            self.log_unnormalised_posterior = np.loadtxt(file,dtype = np.float64)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'log_unnormalised_posterior_mesh_{param_names}.txt'
            self.parameter_mesh = np.loadtxt(file,dtype = np.float64)
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
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Load from txt file
        try:
            file = inference_filename+f'posterior_predictive_mean.txt'
            self.posterior_predictive_mean = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'posterior_predictive_std.txt'
            self.posterior_predictive_std = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(inference_filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = inference_filename+f'posterior_predictive_x.txt'
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
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export log_unnormalised_posterior
        if len(self.log_unnormalised_posterior.shape) == 2:
            # Get parameter names
            param_names = "_".join([str(k).replace("$","").replace("\\","") for k in list(self.inference_metadata['inference']['true_posterior'])[0:self.num_learning_parameters] ])
            # Save to txt file
            np.savetxt((inference_filename+f'log_unnormalised_posterior_{param_names}.txt'),self.log_unnormalised_posterior)
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'log_unnormalised_posterior_{param_names}.txt')}")

            # Save to txt file
            with open((inference_filename+f'log_unnormalised_posterior_mesh_{param_names}.txt'), 'w') as outfile:
                for data_slice in self.parameter_mesh:
                    np.savetxt(outfile, data_slice, fmt='%-7.10f')
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'log_unnormalised_posterior_mesh_{param_names}.txt')}")


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
            # num_params = fundamental_diagram.num_learning_parameters
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
            # parameter_indices = list(itertools.combinations(range(0,fundamental_diagram.num_learning_parameters), 2))
            # parameter_names = list(itertools.combinations(self.parameter_names, 2))
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
            #     Q_hat = np.sum(Q_hat,axis=list(set(range(0,fundamental_diagram.num_learning_parameters)) - set(index))[0])
            #     # Get parameter names
            #     param_names = "_".join([str(p).replace("$","").replace("\\","") for p in [parameter_names[i][index[0]],parameter_names[i][index[1]]]])
            #
            #     # Save to txt file
            #     np.savetxt((inference_filename+f'_log_unnormalised_posterior_{param_names}.txt'),self.log_unnormalised_posterior)
            #     print(f"File exported to {(inference_filename+f'_log_unnormalised_posterior_{param_names}.txt')}")

        elif len(self.log_unnormalised_posterior.shape) < 2:
            raise ValueError(f'Log unnormalised posterior has shape {len(self.log_unnormalised_posterior.shape)} < 2')


    def export_mcmc_samples(self,**kwargs):

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        if hasattr(self,'theta'):
            # Export theta
            # Save to txt file
            np.savetxt((inference_filename+'theta.txt'),self.theta)
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+'theta.txt')}")

        if hasattr(self,'theta_proposed'):
            # Export theta_proposed
            # Save to txt file
            np.savetxt((inference_filename+'theta_proposed.txt'),self.theta_proposed)
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+'theta_proposed.txt')}")

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)


        if hasattr(self,'thermodynamic_integration_theta'):
            # Export thermodynamic integration theta
            # Save to txt file
            with open((inference_filename+f'thermodynamic_integration_theta.txt'), 'w') as outfile:
                for data_slice in self.thermodynamic_integration_theta:
                    np.savetxt(outfile, data_slice, fmt='%-7.10f')
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+'thermodynamic_integration_theta.txt')}")


    def export_posterior_predictive(self,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export posterior_predictive_mean
        # Save to txt file
        np.savetxt((inference_filename+'posterior_predictive_mean.txt'),self.posterior_predictive_mean)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'posterior_predictive_mean.txt')}")

        # Export posterior_predictive_std
        # Save to txt file
        np.savetxt((inference_filename+'posterior_predictive_std.txt'),self.posterior_predictive_std)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'posterior_predictive_std.txt')}")

        # Export posterior_predictive_x
        # Save to txt file
        np.savetxt((inference_filename+'posterior_predictive_x.txt'),self.posterior_predictive_x)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'posterior_predictive_x.txt')}")

    def export_metadata(self,**kwargs):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['inference_metadata'])

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # If metadata exists - import it
        inference_metadata = None
        if os.path.exists((inference_filename+'metadata.json')):
            #  Import metadata where acceptance is part of metadata
            with open((inference_filename+'metadata.json')) as json_file:
                inference_metadata = json.load(json_file)

        # Update results if there are any
        if (inference_metadata is not None and "results" in inference_metadata) and bool(self.inference_metadata and "results" in self.inference_metadata):
            # Update results
            temp_inference_metadata = copy.deepcopy(self.__inference_metadata['results'])
            utils.update(inference_metadata['results'],temp_inference_metadata)
            # print(json.dumps(inference_metadata['results'],indent=2))

            # Update results
            self.__inference_metadata['results'] = utils.update(self.__inference_metadata['results'],inference_metadata['results'])
            # print(json.dumps(self.inference_metadata['results'],indent=2))
        # Add results if there aren't any
        elif (inference_metadata is not None and "results" in inference_metadata) and (bool(self.inference_metadata) or "results" not in self.inference_metadata):
            self.__inference_metadata['results'] = inference_metadata['results']


        #  Export metadata where acceptance is part of metadata
        with open((inference_filename+'metadata.json'), 'w') as outfile:
            json.dump(self.inference_metadata, outfile)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(inference_filename+'metadata.txt')}")


    def export_posterior_plots(self,figs,inference_filename,plot_type,**kwargs):

        # Make sure figs is not empty
        if not hasattr(figs,'__len__') or len(figs) < 1 or not all([bool(v) for v in figs]):
            raise ValueError(f'No figures found in {figs}')

        # Loop through each plot and export it
        for i,f in enumerate(figs):
            # Get parameters in string format separated by _
            param_names = "_".join([str(p).replace("$","").replace("\\","") for p in figs[i]['parameters']])
            # Export plot to file
            figs[i]['figure'].savefig((inference_filename+f'{plot_type}_{param_names}.png'),dpi=300)
            # Close plot
            plt.close(figs[i]['figure'])

            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(inference_filename+f'{plot_type}_{param_names}.png')}")



    def export_univariate_prior_plots(self,fundamental_diagram,show_plot:bool=False,prints:bool=False):

        # Get prior plots
        fig = self.generate_univariate_prior_plots(fundamental_diagram,show_plot,prints)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(fig,inference_filename,'prior')


    def export_log_unnormalised_posterior_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_log_unnormalised_posteriors_plots(fundamental_diagram,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"log_unnormalised_posterior")


    def export_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_mixing_plots(fundamental_diagram,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"mixing")

    def export_thermodynamic_integration_mcmc_mixing_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_thermodynamic_integration_mcmc_mixing_plots(fundamental_diagram,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"mixing")

    def export_mcmc_parameter_posterior_plots(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_parameter_posterior_plots(fundamental_diagram,num_stds,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"parameter_posterior")

    def export_thermodynamic_integration_mcmc_parameter_posterior_plots(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Get subplots
        figs = self.generate_thermodynamic_integration_mcmc_parameter_posterior_plots(fundamental_diagram,num_stds,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"parameter_posterior")

    def export_mcmc_acf_plots(self,fundamental_diagram,show_plot:bool=False):

        # Get subplots
        figs = self.generate_mcmc_acf_plots(fundamental_diagram,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"acf")


    def export_mcmc_space_exploration_plots(self,fundamental_diagram,show_plot:bool=False):

        # Set show posterior plot to true iff the metadata says so AND you have already computed the posterior
        show_posterior = strtobool(self.inference_metadata['plot']['vanilla_mcmc']['include_posterior']) and utils.has_attributes(self,['log_unnormalised_posterior','parameter_mesh'])

        # Generate plots
        figs = self.generate_mcmc_space_exploration_plots(fundamental_diagram,show_posterior,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"space_exploration")

    def export_thermodynamic_integration_mcmc_space_exploration_plots(self,fundamental_diagram,show_plot:bool=False):

        # Set show posterior plot to true iff the metadata says so AND you have already computed the posterior
        show_posterior = strtobool(self.inference_metadata['plot']['thermodynamic_integration_mcmc']['include_posterior']) and utils.has_attributes(self,['log_unnormalised_posterior','parameter_mesh'])

        # Generate plots
        figs = self.generate_thermodynamic_integration_mcmc_space_exploration_plots(fundamental_diagram,show_posterior,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"space_exploration")


    def export_mcmc_posterior_predictive_plot(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False):

        # Generate plots
        figs = self.generate_posterior_predictive_plot(fundamental_diagram,num_stds,show_plot)

        # Get inference filename
        inference_filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_posterior_plots(figs,inference_filename,"posterior_predictive")
