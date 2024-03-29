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
import pandas as pd
import scipy.optimize
import matplotlib
import matplotlib.pyplot as plt
import multiprocessing as mp
import statsmodels.api as sm

from tqdm import tqdm
from functools import partial
from scipy.special import comb
from scipy.optimize import fmin,fsolve,broyden1,broyden2
from distutils.util import strtobool
from probability_distributions import *
from . import temperature_schedules as tp
from pathos.multiprocessing import ProcessingPool

# matplotlib settings
matplotlib.rc('font', **{'size' : 16})

# Define list of latex characters present in parameter names
latex_characters = ["$","\\","^","}","{"]
UNCOMMENT = False
temperature_of_interest = 10

class MarkovChainMonteCarlo(object):

    def __init__(self,inference_id):
        self.inference_id = inference_id

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
    def log_kernel(self):
        return self.__log_kernel

    @log_kernel.setter
    def log_kernel(self,log_kernel):
        self.__log_kernel = log_kernel

    @property
    def log_ti_kernel(self):
        return self.__log_ti_kernel

    @log_ti_kernel.setter
    def log_ti_kernel(self,log_ti_kernel):
        self.__log_ti_kernel = log_ti_kernel

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

    @theta.deleter
    def theta(self):
        del self.__theta

    @property
    def theta_proposed(self):
        return self.__theta_proposed

    @theta_proposed.setter
    def theta_proposed(self,theta_proposed):
        self.__theta_proposed = theta_proposed

    @theta_proposed.deleter
    def theta_proposed(self):
        del self.__theta_proposed

    @property
    def parameter_delta_function(self):
        return self.__parameter_delta_function

    @parameter_delta_function.setter
    def parameter_delta_function(self,parameter_delta_function):
        self.__parameter_delta_function = parameter_delta_function

    @property
    def thermodynamic_integration_theta(self):
        return self.__thermodynamic_integration_theta

    @thermodynamic_integration_theta.deleter
    def thermodynamic_integration_theta(self):
        del self.__thermodynamic_integration_theta

    @thermodynamic_integration_theta.setter
    def thermodynamic_integration_theta(self,thermodynamic_integration_theta):
        self.__thermodynamic_integration_theta = thermodynamic_integration_theta

    @property
    def thermodynamic_integration_theta_proposed(self):
        return self.__thermodynamic_integration_theta_proposed

    @thermodynamic_integration_theta_proposed.deleter
    def thermodynamic_integration_theta_proposed(self):
        del self.__thermodynamic_integration_theta_proposed

    @thermodynamic_integration_theta_proposed.setter
    def thermodynamic_integration_theta_proposed(self,thermodynamic_integration_theta_proposed):
        self.__thermodynamic_integration_theta_proposed = thermodynamic_integration_theta_proposed

    def valid_input(self):

        # Flag for proceeding with experiments
        proceed = True

        vanilla_N = max(int(self.inference_metadata['inference']['vanilla_mcmc']['N']),1)
        vanilla_burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])
        vanilla_posterior_predictive_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['posterior_predictive_samples'])
        vanilla_marginal_likelihood_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['marginal_likelihood_samples'])
        ti_N = max(int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),1)
        ti_burnin = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])
        ti_marginal_likelihood_samples = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['marginal_likelihood_samples'])
        lower_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['lower_bounds'])]
        upper_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['upper_bounds'])]
        parallel_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['parallel_chains'])
        convergence_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['convergence_chains'])
        vanilla_proposal_stds = list(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['proposal_stds'])
        ti_proposal_stds = list(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['proposal_stds'])
        p0 = list(self.inference_metadata['inference']['initialisation']['p0'])
        burnin_step = int(self.inference_metadata['inference']['convergence_diagnostic']['burnin_step'])
        priors = list(self.inference_metadata['inference']['priors'].keys())

        # Sanity checks
        if len(lower_bounds) != len(upper_bounds):
            proceed = False
            print(f'Length of lower and upper bound lists is not equal {len(lower_bounds)} != {len(upper_bounds)}')

        if any([lower_bounds[i]>=upper_bounds[i] for i in range(len(upper_bounds))]):
            proceed = False
            print(f'Upper bounds >= lower bounds in transition kernel')

        if not bool(strtobool(self.inference_metadata['learn_noise'])) and not bool(strtobool(self.simulation_metadata['simulation_flag'])):
            proceed = False
            print(f"Sigma cannot be known if the data is not a simulation")

        if self.inference_metadata['data_id'] != self.simulation_metadata['id']:
            proceed = False
            print(f"Data id records inconsistent: Inference metadata: {self.inference_metadata['data_id']}. Simulation metadata: {self.simulation_metadata['id']}")

        if vanilla_burnin >= vanilla_N:
            proceed = False
            print(f"Vanilla MCMC burnin larger than N {vanilla_burnin} >= {vanilla_N}")

        if burnin_step >= vanilla_N:
            proceed = False
            print(f"Gelman & Rubin criterion burnin step larger than vanilla MCMC N {burnin_step} >= {vanilla_N}")

        if vanilla_posterior_predictive_samples >= (vanilla_N+vanilla_burnin):
            proceed = False
            print(f"Posterior predictive samples exceed N + burnin {vanilla_posterior_predictive_samples} >= {vanilla_N+vanilla_burnin}")

        if vanilla_marginal_likelihood_samples >= (vanilla_N+vanilla_burnin):
            proceed = False
            print(f"Posterior harmonic mean estimator samples exceed N + burnin {vanilla_marginal_likelihood_samples} >= {vanilla_N+vanilla_burnin}")

        if ti_burnin >= ti_N:
            proceed = False
            print(f"Thermodynamic Integration MCMC burnin larger than N {ti_burnin} >= {ti_N}")

        if burnin_step >= ti_N:
            proceed = False
            print(f"Gelman & Rubin criterion burnin step larger than Thermodynamic Integration MCMC N {burnin_step} >= {ti_N}")

        if any([len(vanilla_proposal_stds) != len(ti_proposal_stds),
            len(ti_proposal_stds) != len(p0),
            len(p0) != len(lower_bounds),
            len(lower_bounds) != len(upper_bounds),
            len(upper_bounds) != len(priors)]):

            proceed = False
            print('Number of parameters is inconsistent')
            print('vanilla_mcmc_proposal_stds',len(vanilla_proposal_stds))
            print('thermodynamic_integration_mcmc_proposal_stds',len(ti_proposal_stds))
            print('p0',len(p0))
            print('lower_bounds',len(lower_bounds))
            print('upper_bounds',len(upper_bounds))
            print('Number of priors',len(priors))

        if ti_marginal_likelihood_samples >= (ti_N+ti_burnin):
            proceed = False
            print(f"Thermodynamic Integration marginal likelihood estimator samples exceed N + burnin {ti_marginal_likelihood_samples} >= {ti_N+ti_burnin}")

        if convergence_chains > parallel_chains:
            proceed = False
            print(f"Number of chains used to assess convergence {convergence_chains} > number of chains run in parallel {parallel_chains}")

        return proceed

    def populate(self,fundamental_diagram):

        # Ensure you provide valid inputs
        if not self.valid_input(): raise ValueError(f"Cannot proceed with inference {self.inference_metadata['id']}")

        # Decide how many parameters to learn
        self.raw_data = not (self.simulation_metadata['simulation_flag']) or (self.inference_metadata['data_fundamental_diagram'] != self.inference_metadata['fundamental_diagram'])
        if self.raw_data:
            print('Raw data (non-simulation)')
            self.num_learning_parameters = fundamental_diagram.num_learning_parameters
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]
            self.learn_noise = True
        elif (not strtobool(self.inference_metadata['learn_noise'])) and (len(fundamental_diagram.true_parameters)>0):
            print('Simulation without sigma learning')
            self.num_learning_parameters = fundamental_diagram.num_learning_parameters-1
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]
            self.true_parameters = fundamental_diagram.true_parameters[0:self.num_learning_parameters]
            self.sigma2 = fundamental_diagram.true_parameters[-1]
            self.learn_noise = False
        else:
            print('Simulation with sigma learning')
            self.num_learning_parameters = fundamental_diagram.num_learning_parameters
            self.true_parameters = fundamental_diagram.true_parameters
            self.parameter_names = fundamental_diagram.parameter_names[0:self.num_learning_parameters]
            self.sigma2 = fundamental_diagram.true_parameters[-1]
            self.learn_noise = True

        print('Number of learning parameters',self.num_learning_parameters)

        # Get lower and upper bounds for parameters
        lower_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['lower_bounds'])]
        upper_bounds = [float(x) for x in list(self.inference_metadata['inference']['parameter_constraints']['upper_bounds'])]

        # Define parameter transformations
        self.transformations = utils.map_name_to_parameter_transformation(priors=self.inference_metadata['inference']['priors'],
                                                                        num_learning_parameters=self.num_learning_parameters,
                                                                        lower_bounds=lower_bounds,
                                                                        upper_bounds=upper_bounds)

        # Update data and parameters in inference model
        self.update_data(fundamental_diagram.rho,fundamental_diagram.log_q,r"$\rho$",r"$\log q$")

        # Update temperature schedule
        self.update_temperature_schedule()

        # Update delta functions with parameter constraints
        self.update_parameter_constraint_delta_functions(fundamental_diagram)

        # Update model likelihood
        self.update_log_likelihood_log_pdf(fundamental_diagram)

        # Update model predictive likelihood
        self.update_predictive_likelihood(fundamental_diagram)

        # Update model priors
        self.update_log_prior_log_pdf()

        # Update model transition kernels for vanilla and thermodynamic integration MCMC
        self.update_transition_kernel(proposal_stds=list(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['proposal_stds'])[0:self.num_learning_parameters],mcmc_type='vanilla_mcmc')
        self.update_transition_kernel(proposal_stds=list(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['proposal_stds'])[0:self.num_learning_parameters],mcmc_type='thermodynamic_integration_mcmc')

        # Add results key to metadata
        self.__inference_metadata['results'] = {}

    def transform_parameters(self,p,inverse:bool=False):
        try:
            assert len(p) == len(self.transformations)
        except:
            raise ValueError(f'Parameter length {len(p)} and transformation list length {len(self.transformations)} do not match.')

        transformed_params = []
        for i in range(len(p)):
            if inverse: transformed_params.append(self.transformations[i]['backward'](p[i]))
            else: transformed_params.append(self.transformations[i]['forward'](p[i]))

        return transformed_params


    def update_data(self,x,y,x_name,y_name):
        self.x = x
        self.log_y = y
        self.x_name = x_name
        self.log_y_name = y_name
        self.n = y.shape[0]

        # Make sure you have more than one data point
        if self.n <= 1:
            raise ValueError(f'Not enough data points: {self.n} <= 1.')

    def update_temperature_schedule(self):
        # Get temperature schedule function
        temperature_schedule_fn = tp.map_name_to_temperature_schedule(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_schedule'])
        # Get number of steps and power
        nsteps = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])])
        power = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_power'])])
        # Update class attribute
        self.temperature_schedule = temperature_schedule_fn(nsteps,power)

    def update_parameter_constraint_delta_functions(self,fundamental_diagram,prints:bool=False):

        # Create list of delta functions
        delta_functions = []
        # Flag for whether constraints exist or not
        constraints_exist = False
        # Count number of constraints
        num_constraints = 0

        if 'implicit' in self.inference_metadata['inference']['parameter_constraints'].keys():
            # Implicit constraints exist
            constraints_exist = True

            # Number of constraints
            C = len(list(self.inference_metadata['inference']['parameter_constraints']['implicit']))

            if prints: print(f'C = {C} implicity constraints')

            # Loop through number of parameters
            for i,k in enumerate(list(self.inference_metadata['inference']['parameter_constraints']['implicit'])):
                # Get constraint
                constraint = self.inference_metadata['inference']['parameter_constraints']['implicit'][k]
                if prints: print('implicit constraint',constraint)

                # Define implicit delta function
                def make_implicit_delta_function(index):
                    # Get LHS and RHS of constraint
                    lhs = utils.map_name_to_variable_or_variable_index(self,constraint['lhs'],latex_characters)
                    rhs = utils.map_name_to_variable_or_variable_index(self,constraint['rhs'],latex_characters)
                    operator = constraint['operator']
                    def _implicit_delta_constraint(p):
                        if isinstance(lhs,int):
                            if isinstance(rhs,float):
                                # print(f'comparing p[{lhs}] with rhs:', self.transformations[lhs][1](p[lhs]), 'with', rhs)
                                return utils.map_operation_symbol_to_binary(symbol=operator,lhs=self.transformations[lhs]['backward'](p[lhs]),rhs=rhs)
                            elif isinstance(rhs,int):
                                # print(f'comparing p[{lhs}] with p[{rhs}]:', self.transformations[lhs][1](p[lhs]), 'with', self.transformations[rhs][1](p[rhs]))
                                return utils.map_operation_symbol_to_binary(symbol=operator,lhs=self.transformations[lhs]['backward'](p[lhs]),rhs=self.transformations[rhs]['backward'](p[rhs]))
                        elif isinstance(lhs,float):
                            if isinstance(rhs,float):
                                # print(f'comparing lhs with rhs:', lhs , 'with', rhs)
                                return utils.map_operation_symbol_to_binary(symbol=operator,lhs=lhs,rhs=rhs)
                            elif isinstance(rhs,int):
                                # print(f'comparing lhs with p[{rhs}]:', lhs , 'with', self.transformations[rhs][1](p[rhs]))
                                return utils.map_operation_symbol_to_binary(symbol=operator,lhs=lhs,rhs=self.transformations[rhs]['backward'](p[rhs]))
                    return _implicit_delta_constraint
                # Append function to list of delta functions
                delta_functions.append(make_implicit_delta_function(i))
                # Increment number of constraints
                num_constraints += 1

        if 'explicit' in self.inference_metadata['inference']['parameter_constraints'].keys():

            # Explicit constraints exist
            constraints_exist = True

            # Number of constraints
            C = len(list(self.inference_metadata['inference']['parameter_constraints']['explicit']))

            if prints: print(f'C = {C} explicit constraints')

            # Loop through number of parameters
            for j,e in enumerate(list(self.inference_metadata['inference']['parameter_constraints']['explicit'])):
                # Get constraint
                constraint = self.inference_metadata['inference']['parameter_constraints']['explicit'][e]
                if prints: print('explicit constraint',constraint)

                # Define explicit delta function
                def make_explicit_delta_function(index):
                    # Get LHS and RHS of constraint
                    constraint_function = utils.map_constraint_name_to_function(fundamental_diagram,
                                                                                str(constraint['name']),
                                                                                **constraint)
                    def _explicit_delta_constraint(p):
                        return int(constraint_function(p=p))

                    return _explicit_delta_constraint
                # Append function to list of delta functions
                delta_functions.append(make_explicit_delta_function(j+num_constraints))

        # Take intersection (product) of all existing constraints or set delta function to 1 if there are no constraints
        if constraints_exist:
            # Define intersection of constraints by defining a product over delta functions
            def _delta(p):
                return np.prod([delta(p) for delta in delta_functions])
        else:
            def _delta(p):
                return 1

        # Update parameter delta function
        self.parameter_delta_function = _delta


    def reflect_proposal(self,pnew,mcmc_type,prints:bool=False):
        # Transform parameters
        pnew = self.transform_parameters(pnew[0:self.num_learning_parameters],True)

        # Find lower and upper bounds
        lower_bound = [float(x) for x in list(self.inference_metadata['inference']['transition_kernel']['lower_bound'])]
        upper_bound = [float(x) for x in list(self.inference_metadata['inference']['transition_kernel']['upper_bound'])]

        # Constrain proposal
        for i in range(len(pnew)):
            if pnew[i] <= lower_bound[i]:
                pnew[i] = 2*lower_bound[i]-pnew[i]
                if prints: print('Reflected off lower bound')
            if pnew[i] >= upper_bound[i]:
                pnew[i] = 2*upper_bound[i]-pnew[i]
                if prints: print('Reflected off upper bound')
        return pnew

    def reject_proposal(self,pnew,mcmc_type):
        # Transform parameters
        pnew = self.transform_parameters(pnew[0:self.num_learning_parameters],True)

        # Find lower and upper bounds
        lower_bound = [float(x) for x in list(self.inference_metadata['inference']['transition_kernel']['lower_bound'])]
        upper_bound = [float(x) for x in list(self.inference_metadata['inference']['transition_kernel']['upper_bound'])]
        # Constrain proposal
        for i in range(len(pnew)):
            if pnew[i] <= lower_bound[i] or pnew[i] >= upper_bound[i]:
                return True
        return pnew


    def propose_new_sample(self,p):
        utils.validate_attribute_existence(self,['transition_kernel'])
        return self.__transition_kernel(p)

    def propose_new_sample_thermodynamic_integration(self,p,t):
        utils.validate_attribute_existence(self,['thermodynamic_integration_transition_kernel'])
        return self.__thermodynamic_integration_transition_kernel(p,t)

    def evaluate_parameter_delta_function(self,p):
        return self.__parameter_delta_function(p)

    def evaluate_log_target(self,p):
        utils.validate_attribute_existence(self,['log_target'])
        return self.__log_target(p)

    def evaluate_log_target_thermodynamic_integration(self,p,t):
        utils.validate_attribute_existence(self,['thermodynamic_integration_log_target'])
        return self.__thermodynamic_integration_log_target(p,t)

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

    def evaluate_log_kernel(self,pnew,pprev):
        utils.validate_attribute_existence(self,['log_kernel'])
        return self.__log_kernel(pnew,pprev)

    def evaluate_log_ti_kernel(self,p1,p2,t):
        utils.validate_attribute_existence(self,['log_ti_kernel'])
        return self.__log_ti_kernel(p1,p2,t)

    def evaluate_thermodynamic_integration_log_posterior(self,p,t):
        return self.evaluate_log_joint_prior(p[t,:]) + self.temperature_schedule[t]*self.evaluate_log_likelihood(p[t,:])

    def acceptance_ratio(self,pnew,pprev):
        # Compute parameter constraints
        delta_new = self.evaluate_parameter_delta_function(self.transform_parameters(pnew,True))

        # If proposed move does not satisfy parameter constraints
        if delta_new == 0:
            return 0, -np.inf, np.inf

        # pnew = [ 0.9258142,3.00755043,3.78737999,5.89096989,3.184542,0.68729564,-4.56678862 ]
        # pprev = [ 0.94638877,3.0001441,3.79171751,5.89457763,3.16743865,0.68095388,-4.58506089 ]

        # Compute log targets
        log_pnew = self.evaluate_log_target(pnew) + self.evaluate_log_kernel(pnew,pprev)
        log_pprev = self.evaluate_log_target(pprev) + self.evaluate_log_kernel(pprev,pnew)

        # print('self.evaluate_log_kernel(pnew,pprev)',self.evaluate_log_kernel(pnew,pprev))
        # print("self.evaluate_log_target(pnew)",self.evaluate_log_target(pnew))
        # print('log_pnew',log_pnew)
        # sys.exit(1)

        # print('somepnew',somepnew)
        # print('somepprev',somepprev)
        # print('self.evaluate_log_kernel(somepnew,somepprev)',self.evaluate_log_kernel(somepnew,somepprev))
        # print('self.evaluate_log_kernel(somepprev,somepnew)',self.evaluate_log_kernel(somepprev,somepnew))
        # print('self.evaluate_log_target(somepnew)',self.evaluate_log_target(somepnew))
        # print('self.evaluate_log_target(somepprev)',self.evaluate_log_target(somepprev))
        # print('self.evaluate_log_joint_prior(somepnew)',self.evaluate_log_joint_prior(somepnew))
        # print('self.evaluate_log_joint_prior(somepprev)',self.evaluate_log_joint_prior(somepprev))

        # print('pnew',pnew)
        # print('pprev',pprev)
        # print('log_pnew',log_pnew,'self.evaluate_log_kernel(pnew,pprev)',self.evaluate_log_kernel(pnew,pprev))
        # # 'prior',self.evaluate_log_joint_prior(pnew),'likelihood',self.evaluate_log_likelihood(pnew),
        # print('log_pprev',log_pprev,'self.evaluate_log_kernel(pprev,pnew)',self.evaluate_log_kernel(pprev,pnew))
        # # 'prior',self.evaluate_log_joint_prior(pprev),'likelihood',self.evaluate_log_likelihood(pprev),
        # print('\n')

        if np.isnan(log_pnew) or np.isnan(log_pprev):
            print('NaN value found')
            print('log_pnew',log_pnew,'pnew',self.transform_parameters(pnew,True),'prior',self.evaluate_log_joint_prior(pnew),'likelihood',self.evaluate_log_likelihood(pnew))
            print('log_pprev',log_pprev,'pprev',self.transform_parameters(pprev,True),'prior',self.evaluate_log_joint_prior(pprev),'likelihood',self.evaluate_log_likelihood(pprev))
            print('\n')
            return 0, log_pnew, log_pprev
            # raise ValueError('NaN value found')

        # If numerator is + infty and denominator - infty
        if (np.isinf(log_pnew) and log_pnew > 0) or (np.isinf(log_pprev) and log_pprev < 0):
            return 1, log_pnew, log_pprev
        # If numerator is - infty and denominator + infty
        elif (np.isinf(log_pprev) and log_pprev > 0) or (np.isinf(log_pnew) and log_pnew < 0):
            return 0, log_pnew, log_pprev

        # Compute log difference
        log_acc = log_pnew - log_pprev

        # If exp floating point exceeded
        # Delta function is equal to 1 (see line 574)
        if log_acc >= 709:
            return 1, log_pnew, log_pprev
        else:
            return min(1,np.exp(log_acc)), log_pnew, log_pprev

    def thermodynamic_integration_acceptance_ratio(self,pnew,pprev,t):
        # Compute parameter constraints
        delta_new = self.evaluate_parameter_delta_function(self.transform_parameters(pnew[t,:],True))

        # If proposed move does not satisfy parameter constraints
        if delta_new == 0:
            return 0, -np.inf, np.inf

        # if t == 29:
        #     pnew[t,:] = [ 0.9258142,3.00755043,3.78737999,5.89096989,3.184542,0.68729564,-4.56678862 ]
        #     pprev[t,:] = [ 0.94638877,3.0001441,3.79171751,5.89457763,3.16743865,0.68095388,-4.58506089 ]

        # a = self.evaluate_log_target_thermodynamic_integration(pnew,t)
        # b = self.evaluate_log_ti_kernel(pnew,pprev,t)
        # c = self.evaluate_log_target_thermodynamic_integration(pprev,t)
        # d = self.evaluate_log_ti_kernel(pprev,pnew,t)
        # print('a',a)
        # print('b',b)
        # print('c',c)
        # print('d',d)
        # Compute log targets
        log_pnew = self.evaluate_log_target_thermodynamic_integration(pnew,t) + self.evaluate_log_ti_kernel(pnew,pprev,t)
        log_pprev = self.evaluate_log_target_thermodynamic_integration(pprev,t) + self.evaluate_log_ti_kernel(pprev,pnew,t)

        # print('t',t)
        # print('pprev',pprev[t,:],'self.evaluate_log_ti_kernel(pprev,pnew,t)',self.evaluate_log_ti_kernel(pprev,pnew,t))
        # print('pnew',pnew[t,:],'self.evaluate_log_ti_kernel(pnew,pprev,t)',self.evaluate_log_ti_kernel(pnew,pprev,t))
        # print('\n')
        # if t > 15:
        #     print("i =",t,"t =",self.temperature_schedule[t])
        #     print('pnew',pnew[t,:])
        #     print('log_pnew',log_pnew)
        #     print('self.evaluate_log_ti_kernel(pnew,pprev,t)',self.evaluate_log_ti_kernel(pnew,pprev,t))
        #     print('pprev',pprev[t,:])
        #     print('log_pprev',log_pprev)
        #     print('self.evaluate_log_ti_kernel(pnew,pprev,t)',self.evaluate_log_ti_kernel(pprev,pnew,t))
        #     print('\n')

        if np.isnan(log_pnew) or np.isnan(log_pprev):#\
            # or np.isnan(pprev).any() or np.isinf(pprev).any():

            # delta_prev = self.evaluate_parameter_delta_function(pprev[t,:])
            # print('NaN value found')
            # print('pnew:',self.transform_parameters(pnew[t,:],True),'constraints satisfied:',bool(delta_new))
            # print('log_pnew',log_pnew,'prior',self.evaluate_log_joint_prior(pnew[t,:]),'likelihood',self.evaluate_log_likelihood(pnew[t,:]))
            # print('pprev:',self.transform_parameters(pprev[t,:],True),'constraints satisfied:',bool(delta_prev))
            # print('log_pprev',log_pprev,'prior',self.evaluate_log_joint_prior(pprev[t,:]),'likelihood',self.evaluate_log_likelihood(pprev[t,:]))
            # print('\n')
            return 0, log_pnew, log_pprev
            # raise ValueError('NaN value found')

        # If numerator is + infty and denominator - infty
        if (np.isinf(log_pnew) and log_pnew > 0) or (np.isinf(log_pprev) and log_pprev < 0):
            print('Inf value found')
            print('pnew:',self.transform_parameters(pnew[t,:],True),'constraints satisfied:',bool(delta_new))
            print('log_pnew',log_pnew,'prior',self.evaluate_log_joint_prior(pnew[t,:]),'likelihood',self.evaluate_log_likelihood(pnew[t,:]))
            print('pprev:',self.transform_parameters(pprev[t,:],True),'constraints satisfied:',bool(delta_new))
            print('log_pprev',log_pprev,'prior',self.evaluate_log_joint_prior(pprev[t,:]),'likelihood',self.evaluate_log_likelihood(pprev[t,:]))
            return 1, log_pnew, log_pprev
        # If numerator is - infty and denominator + infty
        elif (np.isinf(log_pprev) and log_pprev > 0) or (np.isinf(log_pnew) and log_pnew < 0):
            return 0, log_pnew, log_pprev

        # Compute log difference
        log_acc = log_pnew - log_pprev

        # If exp floating point exceeded
        # # Delta function is equal to 1 (see line 574)
        if log_acc >= 709:
            return 1, log_pnew, log_pprev
        else:
            return min(1,np.exp(log_acc)), log_pnew, log_pprev


    def vanilla_mcmc(self,i,seed:int=None,prints:bool=False):
        """Vanilla MCMC method for sampling from pdf defined by log_function

        Parameters
        ----------
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
        # utils.validate_parameter_existence(['proposal_stds','beta_step'],self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel'])

        p0 = None
        # Read p0 or randomly initialise it from prior
        if utils.has_parameters(['p0'],self.inference_metadata['inference']['initialisation']) and self.inference_metadata['inference']['initialisation']['param_initialisation'] == 'metadata':
            # Read p0 from toml parameters
            p0 = np.array(self.inference_metadata['inference']['initialisation']['p0'])
            # Print
            if prints: print('p0',p0)
            # Transform p0
            p0 = self.transform_parameters(p0[0:self.num_learning_parameters],False)
        elif utils.has_attributes(self,['map_params']) and self.inference_metadata['inference']['initialisation']['param_initialisation'] == 'map':
            # Use MLE estimate
            p0 = np.array(self.map_params)
            # Print
            if prints: print('p0',self.transform_parameters(p0,True))
            # Update metadata
            utils.update(self.__inference_metadata['inference']['vanilla_mcmc'],{'p0':list(p0)})
        else:
            print('vanilla_mcmc sample_from_univariate_priors needs testing')
            # Sample from prior distributions
            p0 = self.sample_from_univariate_priors(1)
            # Print
            if prints: print('p0',self.transform_parameters(p0,True))
            # Update metadata
            utils.update(self.__inference_metadata['inference']['vanilla_mcmc'],{'p0':p0})
            # Transform p0
            p0 = self.transform_parameters(p0[0:self.num_learning_parameters],False)

        # Initialise output variables
        theta = []
        theta_proposed = []
        acc = 0

        # Store necessary parameters
        p_prev = p0
        # Store number of iterations
        N = max(int(self.inference_metadata['inference']['vanilla_mcmc']['N']),1)
        burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])

        # Flag for adapting proposal stds during burnin
        adapt_proposal_during_burnin = False
        if "adapt_proposal_during_burnin" in self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']:
            adapt_proposal_during_burnin = bool(strtobool(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['adapt_proposal_during_burnin']))
        # print("adapt_proposal_during_burnin",adapt_proposal_during_burnin)

        if prints: print(f'Running MCMC with {N} iterations')

        map_log_target = -1e9
        map = np.ones((len(p0)))*1e9
        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Propose new sample
            p_new = self.propose_new_sample(p_prev)

            # If rejection scheme for constrained theta applies
            if self.inference_metadata['inference']['transition_kernel']['action'] == 'reflect':
                # If theta is NOT within lower and upper bounds reject sample
                p_new = self.reflect_proposal(p_new,'vanilla_mcmc')

            # Calculate acceptance probability
            # if self.inference_metadata['inference']['transition_kernel']['action'] == 'reject':
            #     # If theta is NOT within lower and upper bounds reject sample
            #     if self.reject_proposal(p_new,'vanilla_mcmc'): acc_ratio = 0
            #     lt_new = self.evaluate_log_target(p_new)
            #     lt_prev = self.evaluate_log_target(p_prev)
            # else:
            acc_ratio,lt_new,lt_prev = self.acceptance_ratio(p_new,p_prev)
            # print('vanilla mcmc acc_ratio',acc_ratio)

            # Update Maximum A posteriori estimate if necessary
            if lt_new >= map_log_target:
                map = p_new
                map_log_target = lt_new

            # Sample from Uniform(0,1)
            u = np.random.random()

            # Printing proposals every 0.1*Nth iteration
            # if prints and (i in [int(j/10*N) for j in range(1,11)]):
            # print('p_prev',self.transform_parameters(p_prev,True),'lt_prev',lt_prev)
            # print('p_new',self.transform_parameters(p_new,True),'lt_new',lt_new)
            # print('\n')
            # sys.exit(1)

            # if any(np.isnan(p_prev)) or any(np.isnan(p_new)):
            #     print('p_prev',self.transform_parameters(p_prev,True))
            #     print('p_new',self.transform_parameters(p_new,True))
            #     sys.exit(1)

            # Add proposal to relevant array
            theta_proposed.append(p_new)

            # Accept/Reject
            # Compare log_alpha and log_u to accept/reject sample
            if acc_ratio >= u:
                if prints and (i in [int(j/10*N) for j in range(1,11)]):
                    # print('p_new =',self.transform_parameters(p_new,True))
                    print('Accepted!')
                    print(f'Acceptance rate {int(100*acc / N)}%')
                # Increment accepted sample count
                acc += 1
                # Append to accepted and proposed sample arrays
                theta.append(p_new)
                # Update last accepted sample
                p_prev = p_new
            else:
                if prints and (i in [int(j/10*N) for j in range(1,11)]):
                    print('Rejected...')
                    print(f'Acceptance rate {int(100*acc / N)}%')
                # Append to accepted and proposed sample arrays
                theta.append(p_prev)

            # Update proposal stds after burnin
            if i == (burnin-1):
                # Get proposal factor adjustment from metadata
                proposal_factor_adjustment = 0.5
                if "proposal_factor_adjustment" in self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']:
                    proposal_factor_adjustment = float(self.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['proposal_factor_adjustment'])
                # Adjust posterior variance during burnin by adjustment factor
                post_std = np.std(theta[0:burnin],axis=0)*proposal_factor_adjustment
                # Adjust proposal if flag is true
                if adapt_proposal_during_burnin:
                    print('adapt_proposal_during_burnin',adapt_proposal_during_burnin)
                    # Update transition kernel
                    self.update_transition_kernel(proposal_stds=list(post_std),mcmc_type='vanilla_mcmc',prints=prints)
                    # Update metadata
                    utils.update(self.__inference_metadata['results'],{"vanilla_mcmc": {"adapted_proposal_stds":list(post_std)}})
                # REMOVE THIS BEFORE FORMAL EXPERIMENTS
                # if prints:
                # post_mu = np.mean(theta[0:burnin],axis=0)
                # post_mu_str = [str(x) for x in post_mu]
                post_std_str = [str(x) for x in post_std]
                # print('Empirical mu during burnin','['+", ".join(post_mu_str)+']')
                print('Empirical std during burnin','['+", ".join(post_std_str)+']')
        # Update class attributes
        self.theta = np.array(theta).reshape((N,self.num_learning_parameters))
        self.theta_proposed = np.array(theta_proposed).reshape((N,self.num_learning_parameters))
        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc": {"acceptance_rate":int(100*(acc / N)),"MAP":list(map),"MAP_log_target":float(map_log_target)}})

        result_summary = {"vanilla_mcmc":{}}
        for i in range(self.num_learning_parameters):
            # Parameter name
            param_name = utils.remove_characters(str(self.parameter_names[i]),latex_characters)
            result_summary['vanilla_mcmc'][param_name] = {"mean":self.transform_parameters(np.mean(self.theta[burnin:,:],axis=0),True)[i],"std":self.transform_parameters(np.std(self.theta[burnin:,:],axis=0),True)[i]}
        # Update metadata on results
        utils.update(self.__inference_metadata['results'],result_summary)

        if prints:
            print(f'Acceptance rate {int(100*(acc / N))}%')
            print('Posterior means', self.transform_parameters(np.mean(self.theta[burnin:,:],axis=0),True))
            post_stds = np.std(self.theta[burnin:,:],axis=0)*proposal_factor_adjustment
            post_std_str = [str(x) for x in post_std]
            print('Posterior stds','['+", ".join(post_std_str)+']')
            print('MAP', self.transform_parameters(map,True))
            print('MAP log target', map_log_target)

        return np.array(theta).reshape((N,self.num_learning_parameters)), int(100*(acc / N))

    def thermodynamic_integration_mcmc(self,i,seed:int=None,prints:bool=False):
        """Thermodynamic integration MCMC method for sampling from pdf defined by log_function

        Parameters
        ----------
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

        """ Thermodynamic integration MCMC method for sampling from pdf defined by log_function
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
        utils.validate_attribute_existence(self,['evaluate_log_target','temperature_schedule'])

        # Make sure you have stored necessary attributes
        utils.validate_parameter_existence(['N'],self.inference_metadata['inference']['thermodynamic_integration_mcmc'])
        # utils.validate_parameter_existence(['K_diagonal','beta_step'],self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel'])

        # Store length of temperature schedule
        t_len = self.temperature_schedule.shape[0]

        p0 = None
        # Read p0 or randomly initialise it from prior
        if utils.has_parameters(['p0'],self.inference_metadata['inference']['initialisation']) and self.inference_metadata['inference']['initialisation']['param_initialisation'] == 'metadata':
            # Use p0 from inference metadata
            p0 = np.array(self.inference_metadata['inference']['initialisation']['p0'])[0:self.num_learning_parameters]
            # Print
            if prints: print('p0',p0)
            # Transform p0
            p0 = self.transform_parameters(p0,False)
            # Repeat n times
            p0 = np.repeat([p0],t_len,axis=0)
        elif utils.has_attributes(self,['map_params']) and self.inference_metadata['inference']['initialisation']['param_initialisation'] == 'map':
            # Use MLE estimate
            p0 = np.array(self.map_params)
            # Print
            if prints: print('p0',self.transform_parameters(p0,True))
            # Update metadata
            utils.update(self.__inference_metadata['inference']['thermodynamic_integration_mcmc'],{'p0':list(p0)})
            # Repeat n times
            p0 = np.repeat([p0],t_len,axis=0)
        else:
            print('thermodynamic_integration_mcmc sample_from_univariate_priors needs testing')
            # Sample from prior distributions
            p0 = self.sample_from_univariate_priors(t_len).reshape((t_len,self.num_learning_parameters))
            # Print
            if prints: print('p0',p0)
            # Transform p0
            p0 = np.array([self.transform_parameters(p0[0:self.num_learning_parameters,t],False) for t in range(len(self.temperature_schedule))])
            # Update metadata
            utils.update(self.__inference_metadata['inference']['thermodynamic_integration_mcmc'],{'p0':list(p0)})

        # Store necessary parameters
        p_prev = p0
        # Store number of iterations
        N = np.max([int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),1])
        burnin = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])
        # Read temperature threshold (if it exists)
        temperature_threshold = 0
        if "temperature_threshold" in self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']:
            temperature_threshold = float(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['temperature_threshold'])

        # Flag for adapting proposal stds during burnin
        adapt_proposal_during_burnin = False
        if "adapt_proposal_during_burnin" in self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']:
            adapt_proposal_during_burnin = strtobool(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['adapt_proposal_during_burnin'])

        # Define distribution over temperatures
        temperature_choice_distribution = "uniform"
        if "temperature_choice_distribution" in self.inference_metadata['inference']['thermodynamic_integration_mcmc']:
            temperature_choice_distribution = str(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temperature_choice_distribution'])

        # Decide how many samples to take from each temperature
        temperature_choice_probabilities = None
        if temperature_choice_distribution.lower() == 'uniform':
            temperature_choice_probabilities = np.ones(t_len) * 1/t_len
        else:
            temperature_choice_probabilities = np.ones(t_len)
            temperature_choice_probabilities[0:temperature_of_interest] = 10000
            temperature_choice_probabilities[temperature_of_interest:t_len] = 30000#40000
            temperature_choice_probabilities = temperature_choice_probabilities/np.sum(temperature_choice_probabilities)
        print(temperature_choice_probabilities)

        # Initialise output variables
        theta = np.zeros((N,t_len,self.num_learning_parameters))
        theta_proposed = np.zeros((N,t_len,self.num_learning_parameters))
        acc,prop = np.zeros(t_len),np.zeros(t_len)

        if prints: print(f'Running Thermodynamic Integration MCMC with {N} iterations')

        # sys.exit(1)
        # print(temperature_distribution*100)
        # Loop through MCMC iterations
        for i in tqdm(range(N)):

            # Select random temperature from schedule
            random_t_index = np.random.choice(a=range(0,t_len),p=temperature_choice_probabilities)
            # np.random.choice(a=[temperature_of_interest,temperature_of_interest-1],p=[0.75,0.25])
            # np.random.choice(a=range(0,t_len),p=temperature_choice_probabilities)
            # np.random.choice(a=[temperature_of_interest,t_len-1],p=[0.5,0.5])
            # temperature_of_interest
            # t_len-1
            # np.random.choice(a=range(0,t_len),p=temperature_choice_probabilities)

            # Copy previously accepted sample
            p_new = copy.deepcopy(p_prev)

            # Copy previously accepted sample
            p_new[random_t_index,:] = self.propose_new_sample_thermodynamic_integration(p_prev[random_t_index,:],random_t_index)

            # If rejection scheme for constrained theta applies
            if self.inference_metadata['inference']['transition_kernel']['action'] == 'reflect':
              # If theta is NOT within lower and upper bounds reject sample
              p_new[random_t_index,:] = self.reflect_proposal(p_new[random_t_index,:],'thermodynamic_integration_mcmc')

            # if self.inference_metadata['inference']['transition_kernel']['action'] == 'reject':
            #     # If theta is NOT within lower and upper bounds reject sample
            #     if self.reject_proposal(p_new[random_t_index,:],'thermodynamic_integration_mcmc'): acc_ratio = 0
            #     lt_new = self.evaluate_log_target_thermodynamic_integration(p_new,random_t_index)
            #     lt_prev = self.evaluate_log_target_thermodynamic_integration(p_prev,random_t_index)
            # else:
            # Calculate acceptance probability
            acc_ratio,lt_new,lt_prev = self.thermodynamic_integration_acceptance_ratio(p_new,p_prev,random_t_index)
            # if random_t_index == 29:
                # print('TI mcmc acc_ratio',acc_ratio)

            # Sample from Uniform(0,1)
            u = np.random.random()

            # Printing proposals every 0.1*Nth iteration
            # if prints and (i in [int(j/10*N) for j in range(1,11)]):
            # print('p_prev',self.transform_parameters(p_prev[random_t_index,:],True),'lt_prev',lt_prev)
            # print('p_new',self.transform_parameters(p_new[random_t_index,:],True),'lt_new',lt_new)
            # print('\n')
            # sys.exit(1)

            # Increment proposal counter
            prop[random_t_index] += 1

            # Add proposal to relevant array
            theta_proposed[i,:,:] = p_new

            # Accept/Reject
            # Compare log_alpha and log_u to accept/reject sample
            if acc_ratio >= u:
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
                # print(f'i = {random_t_index}, t_i = {self.temperature_schedule[random_t_index]}, p_new = {self.transform_parameters(p_new[random_t_index,:],True)}')
                print(f'Total acceptance rate {int(100*np.sum(acc) / np.sum(prop))}%')
                # UNCOMMENT THIS
                if UNCOMMENT:
                    if temperature_threshold != 0 and temperature_threshold != 1:
                        high_temp_acceptance = int(100*(np.sum(acc[self.temperature_schedule>temperature_threshold]) / np.sum(prop[self.temperature_schedule>temperature_threshold])))
                        low_temp_acceptance = int(100*(np.sum(acc[self.temperature_schedule<=temperature_threshold]) / np.sum(prop[self.temperature_schedule<=temperature_threshold])))
                        print(f'Low temperature acceptance rate {low_temp_acceptance}%, High temperature acceptance rate {high_temp_acceptance}%')


            # Update proposal stds after burnin
            if i == (burnin-1):
                # Get proposal factor adjustment from metadata
                proposal_factor_adjustment = 0.5
                if "proposal_factor_adjustment" in self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']:
                    proposal_factor_adjustment = float(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['transition_kernel']['proposal_factor_adjustment'])
                # Adjust posterior variance during burnin by adjustment factor
                post_std = np.std(theta[0:burnin,:,:],axis=(0,1))*proposal_factor_adjustment
                # Adjust proposal if flag is true
                if adapt_proposal_during_burnin:
                    # Update transition kernel
                    self.update_transition_kernel(proposal_stds=list(post_std),mcmc_type='thermodynamic_integration_mcmc',prints=prints)
                    # Update metadata
                    utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc": {"adapted_proposal_stds":list(post_std)}})
                # if prints:
                # REMOVE THIS BEFORE FORMAL EXPERIMENTS
                    # post_mu = np.mean(theta[0:burnin,:,:],axis=(0,1))
                    # post_mu_str = [str(x) for x in post_mu]
                post_std_str = [str(x) for x in post_std]
                    # print('Empirical mu during burnin','['+", ".join(post_mu_str)+']')
                print('Empirical std during burnin','['+", ".join(post_std_str)+']')

        # Update class attributes
        self.thermodynamic_integration_theta = np.array(theta).reshape((N,t_len,self.num_learning_parameters))
        self.thermodynamic_integration_theta_proposed = np.array(theta_proposed).reshape((N,t_len,self.num_learning_parameters))

        # Print posterior stds for temperature of interest - DELETE WHEN CLEANING CODE
        a = np.std(self.thermodynamic_integration_theta,axis=0)[temperature_of_interest,:]
        b = "["+ ",".join(str(e) for e in a)+ "]"
        print(b)

        # Compute acceptance rate for temperatures below and above threshold
        # UNCOMMENT THIS
        if UNCOMMENT:
            if temperature_threshold != 0 and temperature_threshold != 1:
                high_temp_acceptance = int(100*(np.sum(acc[self.temperature_schedule>temperature_threshold]) / np.sum(prop[self.temperature_schedule>temperature_threshold])))
                low_temp_acceptance = int(100*(np.sum(acc[self.temperature_schedule<=temperature_threshold]) / np.sum(prop[self.temperature_schedule<=temperature_threshold])))
                utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc": {"high_temp_acceptance_rate":high_temp_acceptance,"low_temp_acceptance_rate":low_temp_acceptance}})

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc": {"acceptance_rate":int(100*(np.sum(acc) / np.sum(prop)))}})
        #"acceptances":list(acc)},"proposals":list(prop)})

        # Store posterior mean and variance for each temperature and parameter
        # result_summary = {"thermodynamic_integration_mcmc":{}}
        # for i in range(self.num_learning_parameters):
        #     # Parameter name
        #     param_name = utils.remove_characters(self.parameter_names[i],latex_characters)
        #     result_summary['thermodynamic_integration_mcmc'][param_name] = {"mean":list(np.mean(self.thermodynamic_integration_theta[:,:,i],axis=0)),"std":list(np.std(self.thermodynamic_integration_theta[:,:,i],axis=0))}
        # # Update metadata on results
        # utils.update(self.__inference_metadata['results'],result_summary)

        if prints:
            print(f'Total acceptance rate {int(100*(np.sum(acc) / np.sum(prop)))}%')
            # UNCOMMENT THIS
            if UNCOMMENT:
                if temperature_threshold != 0 and temperature_threshold != 1:
                    print(f'Low temperature acceptance rate {low_temp_acceptance}%, High temperature acceptance rate {high_temp_acceptance}%')
                print(f"Temperature acceptance rate {[(str(int(100*a))+'%') for a in acc/prop]}")

        return np.array(theta).reshape((N,t_len,self.num_learning_parameters)), int(100*(np.sum(acc) / np.sum(prop)))

    # @staticmethod
    def run_parallel_mcmc(self,type,n:int=None,prints:bool=False):

        # Time execution
        tic = time.perf_counter()

        if 'vanilla_mcmc' in type:
            # Define partial function
            mcmc_partial = partial(self.vanilla_mcmc,seed=None,prints=False)
            # Get number of chains from metadata
            if n is None:
                n_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['parallel_chains'])
            else:
                n_chains = int(n)
        elif 'thermodynamic_integration_mcmc' in type:
            # Define partial function
            mcmc_partial = partial(self.thermodynamic_integration_mcmc,seed=None,prints=False)
            # Get number of chains from metadata
            if n is None:
                n_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['parallel_chains'])
            else:
                n_chains = int(n)
        else:
            raise ValueError(f'MCMC type f{type} not found and therefore cannot run MCMC in parallel.')

        if prints: print(f'Running {n_chains} {type.capitalize().replace("_"," ")} chains in parallel')

        # Initialise multiprocessing pool using pathos library
        pool = ProcessingPool(n_chains)

        try:
            # Map indices to MCMC function
            mcmc_chains = list(pool.map(mcmc_partial,range(n_chains)))
        except:
            pool = None
            pool = ProcessingPool(M)
            # Restart pool
            # pool.restart()
            # Map indices to MCMC function
            mcmc_chains = list(pool.map(mcmc_partial,range(n_chains)))

        pool.close()
        pool.join()
        pool.clear()

        # Print time execution
        if prints:
            for i in range(n_chains):
                print(f'Chain {i} acceptance: {mcmc_chains[i][1]}%')
            toc = time.perf_counter()
            print(f"Run parallel vanilla MCMC chains in {toc - tic:0.4f} seconds")

        return np.array([chain[0] for chain in mcmc_chains]),np.array([chain[1] for chain in mcmc_chains])


    def compute_maximum_a_posteriori_estimate(self,prints:bool=False):

        warnings.simplefilter("ignore")

        # Make sure you have imported metadata
        utils.validate_attribute_existence(self,['inference_metadata','evaluate_log_target','num_learning_parameters'])

        # Fix random seed
        if self.simulation_metadata['seed'] == '' or self.simulation_metadata['seed'].lower() == 'none':
            np.random.seed(None)
        else:
            np.random.seed(int(self.simulation_metadata['seed']))

        # Get p0 from metadata
        p0 = list(self.inference_metadata['inference']['initialisation']['p0'])[0:self.num_learning_parameters]

        # Transform parameter
        p0 = self.transform_parameters(p0,False)


        # If no p0 is provided set p0 to true parameter values
        if len(p0) < 1:
            if self.raw_data:
                raise ValueError('True parameters cannot be found for MAP estimator initialisation.')
            else:
                p0 = self.transform_parameters(self.true_parameters,False)

        # Define negative of log likelihood
        def negative_log_target(p):
            return (-1)*self.evaluate_log_target(p)

        # def positive_alpha_constraint(p):
        #     p = self.transform_parameters(p,True)
        #     return p[0] - p[3]*((p[2]-p[1])/(p[1]*p[2]))**p[4]

        # Optimise parameters for negative_log_target
        map = fmin(negative_log_target, p0, disp = False)[0:self.num_learning_parameters]
        # map = scipy.optimize.minimize(negative_log_target,
        #                     p0,
        #                     constraints = [{'type': 'ineq', 'fun': positive_alpha_constraint}],
        #                     method = 'SLSQP',
        #                     options = {"disp" : False})
        # print('map',map)

        # Get parameters
        self.map_params = map #map.x
        # Evaluate log target
        map_log_target = self.evaluate_log_target(self.map_params)
        map_log_likelihood = self.evaluate_log_likelihood(self.map_params)
        map_log_prior = self.evaluate_log_joint_prior(self.map_params)
        # Evaluate log target for true parameters
        if hasattr(self,'true_parameters'):
            true_log_likelihood = self.evaluate_log_likelihood(self.transform_parameters(self.true_parameters,False))
            true_log_target = self.evaluate_log_target(self.transform_parameters(self.true_parameters,False))
            true_log_prior = self.evaluate_log_joint_prior(self.transform_parameters(self.true_parameters,False))

        # Update class variables
        if hasattr(self,'true_parameters'):
            utils.update(self.__inference_metadata['results'],{"map":{"params":list(self.map_params),"map_log_target":map_log_target,"true_log_target":true_log_target}})
        else:
            utils.update(self.__inference_metadata['results'],{"map":{"params":list(self.map_params),"map":map_log_target}})

        if prints:
            # my_choice = [7,13,0.026]
            print(f'MAP parameters: {self.transform_parameters(self.map_params,True)}')
            print(f'MAP log target: {map_log_target}')
            print(f'MAP log likelihood: {map_log_likelihood}')
            print(f'MAP log prior: {map_log_prior}')
            # print('My choice',my_choice)
            # print(f'My choice log target: {self.evaluate_log_target(self.transform_parameters(my_choice,False))}')
            if hasattr(self,'true_parameters'):
                print(f'True log target: {true_log_target}')
                print(f'True log likelihood: {true_log_likelihood}')
                print(f'True log prior: {true_log_prior}')

    def compute_log_posterior_harmonic_mean_estimator(self,theta_list,prints:bool=False):

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        n_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['marginal_likelihood_samples'])

        # Get number of chain iterations and number of chains
        M,N,P = theta_list.shape

        # Get last n_samples
        theta_list = theta_list[:,(N-n_samples):N,:]

        if prints: print(f'Posterior harmonic mean estimator with M = {M}, N = {n_samples}, P = {P}')

        # Define function for computing log marginal likelihood
        def posterior_harmonic_mean_lml(i):
            # Compute log marginal likelihood
            ml = n_samples * ( np.sum([np.exp(self.evaluate_log_likelihood(theta_list[i,j,:]))**(-1) for j in range(0,n_samples)]) )**(-1)
            lml = np.log(ml)
            return lml

        # Compute posterior harmonic mean estimator of the log marginal likelihood in parallel
        pool = ProcessingPool(M)

        log_marginal_likelihoods = None
        try:
            # Map indices to MCMC function
            log_marginal_likelihoods = list(pool.map(posterior_harmonic_mean_lml,range(M)))
        except:
            pool = None
            pool = ProcessingPool(M)
            # Restart pool
            # pool.restart()
            # Map indices to MCMC function
            log_marginal_likelihoods = list(pool.map(posterior_harmonic_mean_lml,range(M)))

        pool.close()
        pool.join()
        pool.clear()

        # Compute Monte Carlo mean and variance for the log marginal likelihood
        log_marginal_likelihoods_mean = np.mean(log_marginal_likelihoods)
        log_marginal_likelihoods_var = np.var(log_marginal_likelihoods)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"log_marginal_likelihoods":list(log_marginal_likelihoods),"log_marginal_likelihoods_mean":float(log_marginal_likelihoods_mean),"log_marginal_likelihoods_var":float(log_marginal_likelihoods_var)}})

        if prints:
            # Print log marginal likelihood
            # print(f'Log marginal likelihoods = {log_marginal_likelihoods}')
            print(f'Log marginal likelihoods: mean = {log_marginal_likelihoods_mean}, var = {log_marginal_likelihoods_var}')
            # Print time execution
            # toc = time.perf_counter()
            # print(f"Computed posterior harmonic mean estimator in {toc - tic:0.4f} seconds")

        return log_marginal_likelihoods

    def compute_thermodynamic_integration_log_marginal_likelihood_estimator(self,theta_list,prints:bool=False):

        # Time execution
        tic = time.perf_counter()

        # Get burnin and acf lags from plot metadata
        n_samples = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['marginal_likelihood_samples'])

        # Get number of chain iterations and number of chains
        M,N,T,P = theta_list.shape

        # Get last n_samples
        theta_list = theta_list[:,(N-n_samples):N,:,:]

        if prints: print(f'Thermodynamic Integration marginal likelihood estimator with M = {M}, N = {n_samples}, P = {P}, T = {T}')

        # Define function for computing log marginal likelihood
        # @staticmethod
        def thermodynamic_integration_lml(index):
            # Initiliase lml
            lml = np.sum([(self.temperature_schedule[t] - self.temperature_schedule[t-1])*(self.evaluate_log_likelihood(theta_list[index,j,t,:]) + self.evaluate_log_likelihood(theta_list[index,j,t-1,:])) for t in range(1,T) for j in range(0,n_samples)])
            lml /= 2*n_samples
            return lml

        # Compute thermodynamic integration marginal likelihood estimator of the log marginal likelihood in parallel
        pool = ProcessingPool(M)#mp.Pool(M)

        log_marginal_likelihoods = []
        try:
            # Map indices to MCMC function
            log_marginal_likelihoods = list(pool.map(thermodynamic_integration_lml,range(M)))
        except:
            print('Error')
            pool = None
            pool = ProcessingPool(M)#mp.Pool(M)
            # Restart pool
            # pool.restart()
            # Map indices to MCMC function
            log_marginal_likelihoods = list(pool.map(thermodynamic_integration_lml,range(M)))

        pool.close()
        pool.join()
        pool.clear()

        # for i in tqdm(range(M)):
            # log_marginal_likelihoods.append(thermodynamic_integration_lml(i))

        # lmls = log_marginal_likelihoods.get()
        # print('lml',lmls)
        # sys.exit(1)

        # Compute Monte Carlo mean and variance for the log marginal likelihood
        log_marginal_likelihoods_mean = np.mean(log_marginal_likelihoods)
        log_marginal_likelihoods_var = np.var(log_marginal_likelihoods)

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"log_marginal_likelihoods":list(log_marginal_likelihoods),"log_marginal_likelihoods_mean":float(log_marginal_likelihoods_mean),"log_marginal_likelihoods_var":float(log_marginal_likelihoods_var)}})

        if prints:
            # Print log marginal likelihood
            # print(f'Log marginal likelihoods = {log_marginal_likelihoods}')
            print(f'Log marginal likelihoods: mean = {log_marginal_likelihoods_mean}, var = {log_marginal_likelihoods_var}')
            # Print time execution
            # toc = time.perf_counter()
            # print(f"Computed thermodynamic integration marginal likelihood estimator in {toc - tic:0.4f} seconds")

        return log_marginal_likelihoods


    def compute_gelman_rubin_statistic_for_vanilla_mcmc(self,theta_list,prints:bool=False):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html and https://github.com/pymc-devs/pymc/blob/master/pymc/diagnostics.py

        # Time execution
        tic = time.perf_counter()

        # Make sure you have the necessary parameters
        utils.validate_parameter_existence(['r_critical'],self.inference_metadata['inference']['convergence_diagnostic'])

        # Get number of chains to assess convergence from metadata
        convergence_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['convergence_chains'])

        # Get R statistic critical value
        r_critical = float(self.inference_metadata['inference']['convergence_diagnostic']['r_critical'])

        # Get sample step from metadata
        sample_step = int(self.inference_metadata['inference']['convergence_diagnostic']['burnin_step'])

        # Get number of samples used to compute marginal likelihod and/or posterior predictive and take max
        posterior_predictive_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['posterior_predictive_samples'])
        marginal_likelihood_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['marginal_likelihood_samples'])
        # Compute maximum used samples - Compute Gelman and Rubin statistics between burnin and N-max_used_samples
        max_used_samples = max(posterior_predictive_samples,marginal_likelihood_samples)

        # Get only first convergence_chains from theta_list
        theta_list = theta_list[0:convergence_chains,:,:]

        # Get number of chain iterations and number of chains
        M,N,P = theta_list.shape

        # Create list of possible burnin times
        possible_burnins = list(range(sample_step,N-max_used_samples,sample_step))

        if prints: print(f'Gelman Rubin convergence criterion with M = {M}, N = {N-max_used_samples}, P = {P}')

        r_stat = np.ones(P)*1e9
        converged = False
        # Loop over possible burnins
        for burnin in possible_burnins:

            if prints: print(f'Checking convergence with burnin = {burnin}')

            # Calculate between-chain variance
            B_over_m = np.sum([(np.mean(theta_list[:,burnin:(N-max_used_samples),:], 1)[j,:] - np.mean(theta_list[:,burnin:(N-max_used_samples),:],(0,1)))**2 for j in range(M)],axis=0) / (M - 1)

            # Calculate within-chain variances
            W = np.sum([(theta_list[i,burnin:(N-max_used_samples),:] - xbar) ** 2 for i,xbar in enumerate(np.mean(theta_list[:,burnin:(N-max_used_samples),:],1))],(0,1)) / (M * (N-max_used_samples-burnin - 1))

            # (over) estimate of variance
            s2 = W * (N-max_used_samples-burnin-1) / (N-max_used_samples-burnin) + B_over_m

            # Pooled posterior variance estimate
            V = s2 + B_over_m / M

            # Calculate PSRF
            r_stat = V / W

            # Print if chains have converged
            if all(r_stat < r_critical):
                if prints:
                    print(r'Vanilla MCMC chains have converged!')
                    print(pd.DataFrame(r_stat))
                    print(f'Run experiment again with burnin = {burnin}')
                # Update metadata
                converged = True
                utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"converged":True,"burnin":int(burnin)}})
                break

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"r_stat":list(r_stat)}})

        if any(r_stat >= r_critical):
            print(r'Vanilla MCMC chains have NOT converged ...')
            print(pd.DataFrame(r_stat))
            # Update metadata
            utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"converged":False,"burnin":int(N)}})

        # Print time execution
        toc = time.perf_counter()
        if prints: print(f"Computed Gelman & Rubin estimator in {toc - tic:0.4f} seconds")
        return r_stat, converged, burnin


    def compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(self,theta_list,prints:bool=False):
        # See more details here: https://pymc-devs.github.io/pymc/modelchecking.html

        # Time execution
        tic = time.perf_counter()

        # Make sure you have the necessary parameters
        utils.validate_parameter_existence(['r_critical'],self.inference_metadata['inference']['convergence_diagnostic'])

        # Get number of chains to assess convergence from metadata
        convergence_chains = int(self.inference_metadata['inference']['convergence_diagnostic']['convergence_chains'])

        # Get R statistic critical value
        r_critical = float(self.inference_metadata['inference']['convergence_diagnostic']['r_critical'])

        # Get sample step from metadata
        sample_step = int(self.inference_metadata['inference']['convergence_diagnostic']['burnin_step'])

        # Get number of samples used to compute marginal likelihod and/or posterior predictive and take max
        marginal_likelihood_samples = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['marginal_likelihood_samples'])
        # Compute maximum used samples - Compute Gelman and Rubin statistics between burnin and N-max_used_samples
        max_used_samples = marginal_likelihood_samples

        # Get only first convergence_chains from theta_list
        theta_list = theta_list[0:convergence_chains,:,:,:]

        # Get number of chain iterations and number of chains
        M,N,T,P = theta_list.shape

        # Create list of possible burnin times
        possible_burnins = list(range(sample_step,N-max_used_samples,sample_step))

        if prints: print(f'Gelman Rubin convergence criterion with M = {M}, N = {N-max_used_samples}, P = {P}, T = {T}')

        r_stat = list(np.ones((T*P))*1e9)
        converged = False
        # Loop over possible burnins
        for burnin in possible_burnins:

            if prints: print(f'Checking convergence with burnin = {burnin}')

            # Calculate between-chain variance
            B_over_m = np.sum([(np.mean(theta_list[:,burnin:(N-max_used_samples),:,:], 1)[j,:] - np.mean(theta_list[:,burnin:(N-max_used_samples),:,:],(0,1)))**2 for j in range(M)],axis=0) / (M - 1)

            # Calculate within-chain variances
            W = np.sum([(theta_list[i,burnin:(N-max_used_samples),:,:] - xbar) ** 2 for i,xbar in enumerate(np.mean(theta_list[:,burnin:(N-max_used_samples),:,:],1))],(0,1)) / (M * (N-max_used_samples-burnin-1))

            # (over) estimate of variance
            s2 = W * (N-max_used_samples-burnin-1) / (N-max_used_samples-burnin) + B_over_m

            # Pooled posterior variance estimate
            V = s2 + B_over_m / M

            # Calculate PSRF
            r_stat = V / W

            # Flatten array
            r_stat = np.array(r_stat).flatten()

            # Decide if convergence was achieved
            # Print if chains have converged
            if all(r_stat < r_critical):
                if prints:
                    print(r'Thermodynamic Integration MCMC chains have converged!')
                    print(pd.DataFrame(r_stat.reshape((T,P))))
                    print(f'Run experiment again with burnin = {burnin}')
                # Update metadata
                converged = True
                utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"converged":True,"burnin":int(burnin)}})
                break


        # Update metadata
        utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"r_stat":list(r_stat)}})

        # If chains have not converged
        if any(r_stat >= r_critical):
            print(r'Thermodynamic Integration MCMC chains have NOT converged ...')
            print(pd.DataFrame(r_stat.reshape((T,P))))
            # Update metadata
            utils.update(self.__inference_metadata['results'],{"thermodynamic_integration_mcmc":{"converged":False,"burnin":int(N)}})


        # Print time execution
        toc = time.perf_counter()
        if prints: print(f"Computed Gelman & Rubin estimator in {toc - tic:0.4f} seconds")
        return r_stat, converged, burnin



    """---------------------------------------------------------------------------Evaluate and generate posterior data/plots-----------------------------------------------------------------------------"""

    def evaluate_posterior_predictive_moments(self,*args,prints:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Time execution
        tic = time.perf_counter()

        # Get number of posterior predictive samples to use and acf lags from plot metadata
        posterior_predictive_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['posterior_predictive_samples'])

        if prints: print(f'Computing posterior predictive moments over {posterior_predictive_samples} samples...')

        # Set posterior predictive range to covariate range
        x = self.x
        # Set posterior predictive x based on args
        if len(args) == 1 and hasattr(args[0],'__len__'):
            x = args[0]

        # Get seed from metadata
        if self.inference_metadata['inference']['vanilla_mcmc']['seed'] in ['','None']: seed = None
        else: seed = int(self.inference_metadata['inference']['vanilla_mcmc']['seed'])

        # Fix random seed
        np.random.seed(seed)

        # Sample from predictive likelihood
        pp = np.array([self.evaluate_predictive_likelihood(self.theta[j,:],x) for j in range(self.theta.shape[0]-posterior_predictive_samples,self.theta.shape[0])])
        # Compute posterior predictive mean
        pp_mean = np.mean(pp,axis=0)
        # Compute posterior predictive standard deviation
        pp_var = np.mean(pp**2,axis=0) - pp_mean**2

        # Update class variables
        self.posterior_predictive_mean = pp_mean
        self.posterior_predictive_std = np.sqrt(pp_var)
        self.posterior_predictive_x = x

        # Compute execution time
        if prints:
            toc = time.perf_counter()
            print(f"Computed posterior predictive in {toc - tic:0.4f} seconds")

    def evaluate_posterior_predictive_r_squared(self,prints:bool=False):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        # Time execution
        tic = time.perf_counter()

        # Get number of posterior predictive samples to use and acf lags from plot metadata
        posterior_predictive_samples = int(self.inference_metadata['inference']['vanilla_mcmc']['posterior_predictive_samples'])

        if prints: print(f'Computing posterior predictive R squared')

        # Compute mean
        log_y_bar = np.mean(self.log_y)
        SS_res = np.sum((self.posterior_predictive_mean-self.log_y)**2)
        SS_tot = np.sum((self.log_y-log_y_bar)**2)
        R2 = 1 - SS_res/SS_tot

        # Update metadata
        utils.update(self.__inference_metadata['results'],{"vanilla_mcmc":{"R2":R2}})

        # Compute execution time
        if prints:
            toc = time.perf_counter()
            print('R2 = ',R2)
            print(f"Computed posterior predictive R squared in {toc - tic:0.4f} seconds")

        return R2

    def generate_univariate_prior_plots(self,show_plot:bool=False,prints:bool=False,show_title:bool=True):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['log_univariate_priors'])

        # Get prior distribution parameters
        prior_params = list(self.inference_metadata['inference']['priors'].values())

        # Get number of iterations of MCMC
        N = max(int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),1)

        figs = []
        prior_limits = []

        # Mean and variance of moment matching approximated priors
        means = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[0] for j,x in enumerate(self.prior_hyperparameters)]
        vars = [taylor_expansion_of_moments(x[0],x[1],1,self.transformations[j]['forward'],self.transformations[j]['jacobian'],self.transformations[j]['hessian'],self.transformations[j]['third_derivative'],self.transformations[j]['name'])[1] for j,x in enumerate(self.prior_hyperparameters)]

        # Read alpha step for scaling of variance
        alpha_step = 1
        # Get kernel parameters
        kernel_params = self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']
        if "alpha_step" in kernel_params:
            alpha_step = float(kernel_params['alpha_step'])

        # Use MC to sample from prior and plot histogram
        mus = [x[0] for x in self.prior_hyperparameters]
        sigma2s = [x[1]**2 for x in self.prior_hyperparameters]
        mc_samples = np.random.multivariate_normal(mean=mus,cov=np.diag(sigma2s),size=300000)
        # Transform samples
        mc_samples = np.array([self.transform_parameters(x,False) for x in mc_samples])
        # Remove nulls
        mask = np.all(np.isnan(mc_samples) | np.isinf(mc_samples), axis=1)
        mc_samples = mc_samples[~mask]
        # Apply constraints on parameters
        mc_samples = np.array([p for p in mc_samples if self.evaluate_parameter_delta_function(self.transform_parameters(p,True))])
        # print(np.shape(mc_samples))

        # Loop through parameter number
        for i in range(self.num_learning_parameters):

            # print(self.parameter_names[i])
            # print(self.inference_metadata['inference']['priors'].keys())
            # Define parameter transformation
            transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['transformation']

            fig = plt.figure(figsize=(10,8))

            # Get max and min parameter from metadata
            pmin = float(self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['min'])
            pmax = float(self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['max'])


            # Define parameter range
            xrange = np.linspace(pmin,pmax,10000)
            lower_limit = float(self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['min_log_prob'])

            # Transform x range
            xrange = self.transformations[i]['forward'](xrange)

            # Evaluate log probability of specified prior
            yrange = self.log_univariate_priors[i](xrange)[0]
            # Evaluate log probability of approximated prior
            yhatrange = univariate_gaussian(xrange,loc=means[i],scale=alpha_step*vars[i],transformation='')

            prior_mean = np.round(self.log_univariate_priors[i](xrange)[1],5)
            prior_std = np.round(self.log_univariate_priors[i](xrange)[2],5)

            # Store distribution and parameter names
            distribution_name = prior_params[i]['distribution'].capitalize()
            parameter_name = self.parameter_names[i]

            # Plot pdf
            plt.plot(xrange,yrange,color='blue',label='logdf')
            plt.plot(xrange,yhatrange[0],color='red',label='approx logdf',linestyle='dashed')
            # Plot histogram
            histogram,_,_ = plt.hist(mc_samples[:,i],bins=30,density=True,color='purple',label='MC samples')

            # Print hyperparameters
            if prints: print(f'Prior hypeparameters for {transformation} {self.parameter_names[i]}:',', '.join(['{}={!r}'.format(k, v) for k, v in hyperparams.items()]))

            # Change x,y limits
            if len(np.where(yrange>=lower_limit)[0]) > 0:
                xmin = xrange[np.min(np.where(yrange>=lower_limit))]
                xmax = xrange[np.max(np.where(yrange>=lower_limit))]
            else:
                xmin = pmin
                xmax = pmax

            # Append x limits to list
            prior_limits.append([xmin,xmax])

            ymin = lower_limit
            # ymax = max(np.max(yrange[np.isfinite(yrange)]),np.max(yhatrange[0][np.isfinite(yhatrange[0])]))
            ymax = max(np.max(yrange[np.isfinite(yrange)]),np.max(yhatrange[0][np.isfinite(yhatrange[0])]),np.max(histogram))

            # print('xmin',xmin)
            # print('xmax',xmax)
            plt.xlim(left=xmin,right=xmax)
            # Change y limit
            plt.ylim(bottom=ymin,top=ymax)

            # Define prior as function
            prior_function = lambda x: self.log_univariate_priors[i](x)[0]
            abs_prior_function = lambda x: abs(self.log_univariate_priors[i](x)[0])
            root1 = fsolve(prior_function,0.0)
            root2 = fmin(abs_prior_function,0.0,disp=False)
            # print('roots',root1,root2)

            # Plot high density region
            plt.vlines(root1,ymin=ymin,ymax=ymax,color='black')#,label='root1')
            plt.vlines(root2,ymin=ymin,ymax=ymax,color='black')#,label='root1')
            plt.hlines(0,xmin=xmin,xmax=xmax,color='black')

            # Plot true parameter if it exists
            if hasattr(self,'true_parameters'):
                plt.vlines(self.transform_parameters(self.true_parameters,False)[i],ymin=ymin,ymax=ymax,color='black',label='Simulation parameter',linestyle='dashed',zorder=5)
            # Plot prior mean
            # plt.vlines(self.transformations[i][0](prior_mean),ymin=ymin,ymax=ymax,color='red',label=f'mean = {prior_mean}',zorder=4)
            # Plot prior mean +/- prior std
            # plt.hlines((ymax+ymin)/2,xmin=(self.transformations[i][0](prior_mean)-prior_std),xmax=(self.transformations[i][0](prior_mean)+prior_std),color='green',label=f'mean +/- std, std = {prior_std}')

            # Set title
            if show_title: plt.title(f"{distribution_name} prior for {transformation} {parameter_name} parameter")
            # Plot legend
            plt.legend(fontsize=10)

            # Xlabel
            plt.xlabel(f"{transformation} {parameter_name}")
            plt.ylabel(f"Log pdf")

            # Plot figure
            if show_plot: plt.show()
            # Append to figures
            figs.append({"parameters":[(utils.remove_characters(transformation,latex_characters)+self.parameter_names[i])],"figure":fig})
            # Close plot
            plt.close(fig)

            # Store prior limits to class attribute
            self.prior_limits = prior_limits


        return figs


    def generate_mcmc_mixing_plots(self,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.num_learning_parameters:
            raise ValueError(f'Posterior has {self.num_learning_parameters} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])

        figs = []
        # Loop through parameter indices
        for p in range(self.num_learning_parameters):
            # Define parameter transformation
            transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[p],latex_characters)]['transformation']
            # Get transformation name
            transformation_name = utils.remove_characters(transformation,latex_characters)

            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Add samples plot
            plt.plot(range(burnin,self.theta.shape[0]),self.theta[burnin:,p],color='blue',label='Samples',zorder=1)

            # Plot true parameters if they exist
            if hasattr(self,'true_parameters') and show_sim_param:
                plt.hlines(self.transform_parameters(self.true_parameters,False)[p],xmin=burnin,xmax=(self.theta.shape[0]),color='black',label='Simulation parameter',zorder=5)

            print(self.parameter_names[p])
            print('Posterior mean',self.transform_parameters(np.mean(self.theta[burnin:,:],axis=0),True)[p])

            # Plot inferred mean
            plt.hlines(np.mean(self.theta[burnin:,p],axis=0),xmin=burnin,xmax=(self.theta.shape[0]),color='red',label=f'Posterior $\mu$',zorder=4)

            # Add labels
            plt.xlabel('MCMC Iterations')
            plt.ylabel(f"{transformation_name} MCMC Samples")
            if show_title: plt.title(f'Mixing for {transformation} {self.parameter_names[p]} with burnin = {burnin}')

            # Add legend
            plt.legend(fontsize=10)

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[(utils.remove_characters(transformation,latex_characters)+self.parameter_names[p])],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_thermodynamic_integration_mcmc_mixing_plots(self,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):
        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.num_learning_parameters:
            raise ValueError(f'Posterior has {self.num_learning_parameters} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])

        figs = []

        # Get number of steps and power
        nsteps = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])])
        power = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_power'])])

        # Loop through parameter indices
        # for ti in range(len(self.temperature_schedule)):
        # REMOVE THIS BEFORE FORMAL EXPERIMENTS
        # Read temperature threshold
        temperature_threshold = 0
        threshold_temperature = 0
        if "temperature_threshold" in self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']:
            temperature_threshold = float(self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']['temperature_threshold'])
            threshold_temperature = next(x[0] for x in enumerate(self.temperature_schedule) if x[1] > temperature_threshold)

        if threshold_temperature == 0:
            # temps = [0,len(self.temperature_schedule)-1]
            temps = list(range(0,len(self.temperature_schedule)-1,5))
            temps.append(len(self.temperature_schedule)-1)
        else:
            temps = [0,threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]
            # [threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]
            #[temperature_of_interest]
            #[0,threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]
        temps = [29]#[temperature_of_interest,29]

        for ti in temps:
        # for ti in range(len(np.concatenate([self.temperature_schedule[0:2],self.temperature_schedule[(len(self.temperature_schedule)-3):len(self.temperature_schedule-1)]]))):

            # Loop through parameter indices
            for p in range(self.num_learning_parameters):
                # Define parameter transformation
                transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[p],latex_characters)]['transformation']
                # Get transformation name
                transformation_name = utils.remove_characters(transformation,latex_characters)

                # Generate figure
                fig = plt.figure(figsize=(10,8))

                # Add samples plot
                plt.plot(range(burnin,self.thermodynamic_integration_theta.shape[0]),self.thermodynamic_integration_theta[burnin:,ti,p],color='blue',label='Samples',zorder=2)

                # Plot true parameters if they exist
                if hasattr(self,'true_parameters') and show_sim_param:
                    plt.hlines(self.transform_parameters(self.true_parameters,False)[p],xmin=burnin,xmax=(self.thermodynamic_integration_theta.shape[0]),label='Simulation parameter',color='black',zorder=3)

                # Plot inferred mean
                plt.hlines(np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0),xmin=burnin,xmax=(self.thermodynamic_integration_theta.shape[0]),color='red',label=f'Posterior $\mu$',linestyle='dashed',zorder=4)

                # Add labels
                plt.xlabel('MCMC Iterations')
                plt.ylabel(f"{transformation_name} MCMC Samples")
                if show_title: plt.title(f'Mixing {transformation} {self.parameter_names[p]}, t = ({ti}/{nsteps-1})^{power}, burnin = {burnin}')

                # Add legend
                plt.legend(fontsize=10)

                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[utils.remove_characters(transformation,latex_characters)+self.parameter_names[p]+("_temperature_")+str(ti)],"figure":fig})
                # Close current plot
                plt.close(fig)

        return figs


    def generate_mcmc_acf_plots(self,show_plot:bool=False,show_title:bool=True):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.num_learning_parameters:
            raise ValueError(f'Posterior has {self.num_learning_parameters} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and acf lags from plot metadata
        burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])
        lags = np.min([int(self.inference_metadata['plot']['acf_lags']),(self.theta[burnin:,:].shape[0]-1)])

        figs = []
        # Loop through parameter indices
        for p in range(self.num_learning_parameters):
            # Define parameter transformation
            transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[p],latex_characters)]['transformation']

            # Generate figure
            fig,ax = plt.subplots(1,figsize=(10,8))

            # Add ACF plot
            if show_title: sm.graphics.tsa.plot_acf(self.theta[burnin:,p], ax=ax, lags=lags, title=f'ACF plot for {transformation} {self.parameter_names[p]} with burnin = {burnin}')
            else: sm.graphics.tsa.plot_acf(self.theta[burnin:,p], ax=ax, lags=lags, title="")

            # Add labels
            ax.set_ylabel(f'{transformation} {self.parameter_names[p]}')
            ax.set_xlabel('Lags')

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[(utils.remove_characters(transformation,latex_characters)+self.parameter_names[p])],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_mcmc_parameter_posterior_plots(self,num_stds:int=2,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.num_learning_parameters:
            raise ValueError(f'Posterior has {self.num_learning_parameters} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and histogram bins from plot metadata
        burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])
        bins = np.max([int(self.inference_metadata['plot']['hist_bins']),10])

        figs = []

        # Compute posterior mean
        posterior_mean = np.mean(self.theta[burnin:,:],axis=0)
        # Compute posterior std
        posterior_std = np.std(self.theta[burnin:,:],axis=0)

        # Loop through parameter indices
        for i in range(self.num_learning_parameters):
            # Define parameter transformation
            transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['transformation']

            # Generate figure
            fig = plt.figure(figsize=(10,8))

            # Plot parameter posterior
            freq,_,_ = plt.hist(self.theta[burnin:,i],bins=bins)

            # Add labels
            if show_title: plt.title(f'Parameter posterior for {transformation} {self.parameter_names[i]} with burnin = {burnin}')
            plt.vlines(posterior_mean[i],0,np.max(freq),color='red',label=r'$\mu$', linewidth=2)
            plt.vlines(posterior_mean[i]-num_stds*posterior_std[i],0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2)
            plt.vlines(posterior_mean[i]+num_stds*posterior_std[i],0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2)
            # Plot true parameters if they exist and flag is true
            if hasattr(self,'true_parameters') and show_sim_param:
                plt.vlines(self.transform_parameters(self.true_parameters,False)[i],0,np.max(freq),label='Simulation parameter',color='black',linewidth=2)
            plt.xlabel(f'{transformation} {self.parameter_names[i]}')
            plt.ylabel('Sample frequency')
            plt.legend(fontsize=10)

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[(utils.remove_characters(transformation,latex_characters)+self.parameter_names[i])],"figure":fig})
            # Close current plot
            plt.close(fig)

        return figs


    def generate_thermodynamic_integration_mcmc_parameter_posterior_plots(self,num_stds:int=2,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):
        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta'])

        # Make sure posterior has right number of parameters
        if self.num_learning_parameters > self.num_learning_parameters:
            raise ValueError(f'Posterior has {self.num_learning_parameters} parameters instead of at least {self.num_learning_parameters}')

        # Get burnin and histogram bins from plot metadata
        burnin = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])
        bins = np.max([int(self.inference_metadata['plot']['hist_bins']),10])

        figs = []

        # Get number of steps and power
        nsteps = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])])
        power = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_power'])])

        # Loop through parameter indices
        for ti in range(len(self.temperature_schedule)):

            # Loop through parameter indices
            for p in range(self.num_learning_parameters):
                # Define parameter transformation
                transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[p],latex_characters)]['transformation']

                # Generate figure
                fig = plt.figure(figsize=(10,8))

                # Plot parameter posterior
                freq,_,_ = plt.hist(self.thermodynamic_integration_theta[burnin:,ti,p],bins=bins,zorder=2)

                # Add labels
                if show_title: plt.title(f'Parameter posterior {transformation} {self.parameter_names[p]}, t = ({ti}/{nsteps-1})^{power}, burnin = {burnin}')
                plt.vlines(np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0),0,np.max(freq),color='red',label=r'$\mu$', linewidth=2,zorder=3)
                plt.vlines((np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)-num_stds*np.std(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)),0,np.max(freq),color='red',label=f'$\mu - {num_stds}\sigma$',linestyle='dashed', linewidth=2,zorder=3)
                plt.vlines((np.mean(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)+num_stds*np.std(self.thermodynamic_integration_theta[burnin:,ti,p],axis=0)),0,np.max(freq),color='red',label=f'$\mu + {num_stds}\sigma$',linestyle='dashed', linewidth=2,zorder=3)
                # Plot true parameters if they exist
                if hasattr(self,'true_parameters') and show_sim_param:
                    plt.vlines(self.transform_parameters(self.true_parameters,False)[p],0,np.max(freq),label='Simulation parameter',color='black',linewidth=2,zorder=4)
                plt.xlabel(f'{transformation} {self.parameter_names[p]}')
                plt.ylabel('Sample frequency')
                plt.legend(fontsize=10)

                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[utils.remove_characters(transformation,latex_characters)+self.parameter_names[p]+("_temperature_")+str(ti)],"figure":fig})
                # Close current plot
                plt.close(fig)

        return figs

    def generate_vanilla_mcmc_space_exploration_plots(self,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['theta','theta_proposed'])

        # Get starting time
        start = time.time()

        # Get number of plots
        num_plots = int(comb(self.num_learning_parameters,2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,self.num_learning_parameters), 2))
        parameter_names = list(itertools.combinations(self.parameter_names, 2))

        # Raise error if asked to plot 0 plots
        if num_plots <= 0:
            raise ValueError(f'You cannot plot {num_plots} plots!')

        # Loop through each plot
        figs = []
        for i in range(num_plots):
            # Get parameter indices
            index = parameter_indices[i]

            # Create figure
            fig = plt.figure(figsize=(10,8))

            # Get burnin
            burnin = int(self.inference_metadata['inference']['vanilla_mcmc']['burnin'])

            # Add samples plot
            plt.scatter(self.theta[burnin:,index[0]],self.theta[burnin:,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=5)
            plt.scatter(self.theta_proposed[burnin:,index[0]],self.theta_proposed[burnin:,index[1]],color='purple',label='Proposed',marker='x',s=50,zorder=4)

            # Define parameter transformation
            transformation_0 = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[index[0]],latex_characters)]['transformation']
            transformation_1 = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[index[1]],latex_characters)]['transformation']

            # Plot true parameters if they exist
            if hasattr(self,'true_parameters') and show_sim_param:
                plt.scatter(self.transform_parameters(self.true_parameters,False)[index[0]],self.transform_parameters(self.true_parameters,False)[index[1]],label='Simulation parameter',marker='x',s=100,color='black',zorder=7)

            # Add labels
            plt.xlabel(f'{transformation_0}{parameter_names[i][0]}')
            plt.ylabel(f'{transformation_1}{parameter_names[i][1]}')
            # Add title
            if show_title: plt.title(f'{transformation_0}{parameter_names[i][0]},{transformation_1}{parameter_names[i][1]} space exploration with burnin = {burnin}')
            # Add legend
            plt.legend(fontsize=10)

            # Show plot
            if show_plot: plt.show()
            # Append plot to list
            figs.append({"parameters":[(utils.remove_characters(transformation_0,latex_characters)+parameter_names[i][0]),
                                    (utils.remove_characters(transformation_1,latex_characters)+parameter_names[i][1])],
                        "figure":fig})
            # Close current plot
            plt.close(fig)

        return figs

    def generate_thermodynamic_integration_mcmc_space_exploration_plots(self,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['thermodynamic_integration_theta','thermodynamic_integration_theta_proposed'])

        # Get starting time
        start = time.time()

        # Get number of plots
        num_plots = int(comb(self.num_learning_parameters,2))

        # Get plot combinations
        parameter_indices = list(itertools.combinations(range(0,self.num_learning_parameters), 2))
        parameter_names = list(itertools.combinations(self.parameter_names, 2))

        # Get burnin
        burnin = int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])

        # Raise error if there are zero plots
        if num_plots <= 0:
            raise ValueError(f'You cannot plot {num_plots} plots!')

        # Loop through each plot
        figs = []

        # Get number of steps and power
        nsteps = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_nsteps'])])
        power = np.max([1,int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['temp_power'])])

        for i in range(num_plots):
            # Get parameter indices
            index = parameter_indices[i]

            # REMOVE THIS BEFORE FORMAL EXPERIMENTS
            # for tj in range(len(self.temperature_schedule)):
            # Read temperature threshold
            temperature_threshold = 0
            threshold_temperature = 0
            if "temperature_threshold" in self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']:
                temperature_threshold = float(self.inference_metadata['inference']["thermodynamic_integration_mcmc"]['transition_kernel']['temperature_threshold'])
                threshold_temperature = next(x[0] for x in enumerate(self.temperature_schedule) if x[1] > temperature_threshold)

            if threshold_temperature == 0:
                # temps = [0,len(self.temperature_schedule)-1]
                temps = list(range(0,len(self.temperature_schedule)-1,5))
                temps.append(len(self.temperature_schedule)-1)
            else:
                temps = [0,threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]
                #[threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]
                #[temperature_of_interest]
                #[0,threshold_temperature-1,threshold_temperature,len(self.temperature_schedule)-1]

            temps = [29]#[temperature_of_interest,29]

            for tj in temps:
                # Define parameter transformation
                transformation_0 = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[index[0]],latex_characters)]['transformation']
                transformation_1 = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[index[1]],latex_characters)]['transformation']

                # Create figure
                fig = plt.figure(figsize=(10,8))

                # Add samples plot
                plt.scatter(self.thermodynamic_integration_theta[burnin:,tj,index[0]],self.thermodynamic_integration_theta[burnin:,tj,index[1]],color='red',label='Accepted',marker='x',s=50,zorder=5)
                plt.scatter(self.thermodynamic_integration_theta_proposed[burnin:,tj,index[0]],self.thermodynamic_integration_theta_proposed[burnin:,tj,index[1]],color='purple',label='Proposed',marker='x',s=50,zorder=4)

                # Plot true parameters if they exist
                if hasattr(self,'true_parameters') and show_sim_param:
                    plt.scatter(self.transform_parameters(self.true_parameters,False)[index[0]],self.transform_parameters(self.true_parameters,False)[index[1]],label='Simulation parameter',marker='x',s=100,color='black',zorder=4)

                # Add labels
                plt.xlabel(f'{transformation_0}{parameter_names[i][0]}')
                plt.ylabel(f'{transformation_1}{parameter_names[i][1]}')
                # Add title
                if show_title: plt.title(f'{transformation_0}{parameter_names[i][0]},{transformation_1}{parameter_names[i][1]} exploration, t = ({tj}/{nsteps-1})^{power} burnin = {burnin}')
                # Add legend
                plt.legend(fontsize=10)

                # Show plot
                if show_plot: plt.show()
                # Append plot to list
                figs.append({"parameters":[utils.remove_characters(transformation_0,latex_characters)+parameter_names[i][0],utils.remove_characters(transformation_1,latex_characters)+parameter_names[i][1],f'temperature_{tj}'],"figure":fig})
                # Close current plot
                plt.close(fig)


        return figs

    def generate_posterior_predictive_plot(self,fundamental_diagram,num_stds:int=2,show_plot:bool=False,show_title:bool=True):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        # Get starting time
        start = time.time()

        figs = []

        # Store flag for whether data should be kept in log scale or transformed
        multiplicative = strtobool(self.simulation_metadata['error']['multiplicative'])

        transformed_parameter_names = []
        for i,param_name in enumerate(self.parameter_names):
            # Define parameter transformation
            transformation = self.inference_metadata['inference']['priors'][utils.remove_characters(self.parameter_names[i],latex_characters)]['transformation']
            param_name = utils.remove_characters(param_name,latex_characters)
            # Append transformed name to list
            transformed_parameter_names.append((transformation+param_name))

        # Create figure
        fig = plt.figure(figsize=(10,8))

        # Compute upper and lower bounds
        q_upper = self.posterior_predictive_mean + num_stds*self.posterior_predictive_std
        q_mean = self.posterior_predictive_mean
        q_lower= self.posterior_predictive_mean - num_stds*self.posterior_predictive_std

        plt.scatter(self.x,self.log_y,label='Observed data',color='blue',zorder=2,s=10)
        plt.plot(self.posterior_predictive_x,q_mean,color='red',label=r'$\mu$',zorder=1)
        if hasattr(self,'true_parameters'):
            if multiplicative:
                q_true = fundamental_diagram.log_simulate_with_x(self.true_parameters,self.posterior_predictive_x)
            else:
                q_true = np.exp(fundamental_diagram.log_simulate_with_x(self.true_parameters,self.posterior_predictive_x))
            plt.plot(fundamental_diagram.rho,q_true,color='black',label='Simulation parameter',zorder=2)
        plt.fill_between(self.posterior_predictive_x,q_upper,q_lower,alpha=0.5,color='red',label=f"$\mu$ +/- {num_stds}$\sigma$",zorder=3)
        if show_title: plt.title(f"Posterior predictive for {self.inference_metadata['fundamental_diagram']} FD")
        plt.xlabel(f'{self.x_name}')
        plt.ylabel(f'{self.log_y_name}')
        plt.legend(fontsize=10)

        # print('q_mean',q_mean)
        # print('q_true',q_true)

        # Show plot
        if show_plot: plt.show()
        # Append plot to list
        figs.append({"parameters":transformed_parameter_names,"figure":fig})
        # Close current plot
        plt.close(fig)

        return figs





    """ ---------------------------------------------------------------------------Import data-----------------------------------------------------------------------------"""


    def import_metadata(self,experiment:str='',prints:bool=False):

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Make sure file exists
        if not os.path.exits((filename+'metadata.json')):
            raise FileNotFoundError(f"Metadata file {filename}metadata.json not found")

        #  Import metadata where acceptance is part of metadata
        with open((filename+'metadata.json')) as json_file:
            self.inference_metadata = json.load(json_file)

        if prints: print('Imported MCMC samples')


    def import_vanilla_mcmc_samples(self,experiment:str='', prints:bool=False):

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Get burnins for Vanilla MCMC and Thermodynamic integration MCMC
        vanilla_burnin = np.max([int(self.inference_metadata['inference']['vanilla_mcmc']['burnin']),0])

        # Load theta from txt file
        file = (filename+f'theta.txt')
        if prints: print('Importing MCMC samples')
        if os.path.exists(file):
            self.theta = np.loadtxt(file)

            # Update posterior mean and std in inference metadata results section
            result_summary = {"vanilla_mcmc":{}}
            for i in range(self.num_learning_parameters):
                # Parameter name
                param_name = utils.remove_characters(self.parameter_names[i],latex_characters)
                result_summary['vanilla_mcmc'][param_name] = {"mean":np.mean(self.theta[vanilla_burnin:,i]),"std":np.std(self.theta[vanilla_burnin:,i])}
            # Update metadata on results
            utils.update(self.__inference_metadata['results'],result_summary)

        if prints: print('Importing Vanilla MCMC proposals')
        # Load theta proposed from txt file
        file = (filename+f'theta_proposed.txt')
        if os.path.exists(file):
            self.theta_proposed = np.loadtxt(file)

        if prints: print('Imported Vanilla MCMC samples')


    def import_thermodynamic_integration_mcmc_samples(self,experiment:str='',prints:bool=False):

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,'thermodynamic_integration',inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Get burnins for Vanilla MCMC and Thermodynamic integration MCMC
        ti_burnin = np.max([int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin']),0])

        if prints: print('Importing Thermodynamic Integration MCMC samples')

        # Load thermodynamic integration theta from txt file
        file = (filename+f'thermodynamic_integration_theta.txt')
        if os.path.exists(file):
            self.thermodynamic_integration_theta = np.loadtxt(file,dtype='float64')
            # Reshape
            self.thermodynamic_integration_theta = self.thermodynamic_integration_theta.reshape((int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),len(self.temperature_schedule),self.num_learning_parameters))

            # Update posterior mean and std in inference metadata results section
            result_summary = {"thermodynamic_integration_mcmc":{}}
            for i in range(self.num_learning_parameters):
                # Parameter name
                param_name = param_name = utils.remove_characters(self.parameter_names[i],latex_characters)
                result_summary['thermodynamic_integration_mcmc'][param_name] = {"mean":list(np.mean(self.thermodynamic_integration_theta[ti_burnin:,:,i],axis=0)),"std":list(np.std(self.thermodynamic_integration_theta[ti_burnin:,:,i],axis=0))}
            # Update metadata on results
            utils.update(self.__inference_metadata['results'],result_summary)

        # Load thermodynamic integration theta from txt file
        file = (filename+f'thermodynamic_integration_theta_proposed.txt')
        if os.path.exists(file):
            self.thermodynamic_integration_theta_proposed = np.loadtxt(file,dtype='float64')
            # Reshape
            self.thermodynamic_integration_theta_proposed = self.thermodynamic_integration_theta_proposed.reshape((int(self.inference_metadata['inference']['thermodynamic_integration_mcmc']['N']),len(self.temperature_schedule),self.num_learning_parameters))



        if prints: print('Imported Thermodynamic Integration MCMC samples')

    def import_posterior_predictive(self,experiment:str='',prints:bool=False):

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Load from txt file
        try:
            file = filename+f'posterior_predictive_mean.txt'
            self.posterior_predictive_mean = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = filename+f'posterior_predictive_std.txt'
            self.posterior_predictive_std = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        # Load from txt file
        try:
            file = filename+f'posterior_predictive_x.txt'
            self.posterior_predictive_x = np.loadtxt(file)
        except:
            print('Available files are',list(glob.glob(filename + "*.txt")))
            raise Exception(f'File {file} was not found.')

        if prints: print('Imported log unnormalised posterior')


    """ ---------------------------------------------------------------------------Export data/plots-----------------------------------------------------------------------------"""

    def export_mcmc_samples(self,experiment:str='',prints:bool=False):

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        if hasattr(self,'theta'):
            # Export theta
            # Save to txt file
            np.savetxt((filename+'theta.txt'),self.theta)
            if prints: print(f"File exported to {(filename+'theta.txt')}")

        if hasattr(self,'theta_proposed'):
            # Export theta_proposed
            # Save to txt file
            np.savetxt((filename+'theta_proposed.txt'),self.theta_proposed)
            if prints: print(f"File exported to {(filename+'theta_proposed.txt')}")

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,'thermodynamic_integration',inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        if hasattr(self,'thermodynamic_integration_theta'):
            # Export thermodynamic integration theta
            # Save to txt file
            with open((filename+f'thermodynamic_integration_theta.txt'), 'w') as outfile:
                for data_slice in self.thermodynamic_integration_theta:
                    np.savetxt(outfile, data_slice, fmt='%-7.10f')
            if prints: print(f"File exported to {(filename+'thermodynamic_integration_theta.txt')}")

        if hasattr(self,'thermodynamic_integration_theta_proposed'):
            # Export thermodynamic integration theta
            # Save to txt file
            with open((filename+f'thermodynamic_integration_theta_proposed.txt'), 'w') as outfile:
                for data_slice in self.thermodynamic_integration_theta_proposed:
                    np.savetxt(outfile, data_slice, fmt='%-7.10f')
            if prints: print(f"File exported to {(filename+'thermodynamic_integration_theta_proposed.txt')}")



    def export_posterior_predictive(self,experiment:str='',prints:bool=False):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['posterior_predictive_mean','posterior_predictive_std','posterior_predictive_x'])

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export posterior_predictive_mean
        # Save to txt file
        np.savetxt((filename+'posterior_predictive_mean.txt'),self.posterior_predictive_mean)
        if prints: print(f"File exported to {(filename+'posterior_predictive_mean.txt')}")

        # Export posterior_predictive_std
        # Save to txt file
        np.savetxt((filename+'posterior_predictive_std.txt'),self.posterior_predictive_std)
        if prints: print(f"File exported to {(filename+'posterior_predictive_std.txt')}")

        # Export posterior_predictive_x
        # Save to txt file
        np.savetxt((filename+'posterior_predictive_x.txt'),self.posterior_predictive_x)
        if prints: print(f"File exported to {(filename+'posterior_predictive_x.txt')}")

    def export_metadata(self,experiment:str='',prints:bool=False):

        # Make sure you have necessary attributes
        utils.validate_attribute_existence(self,['inference_metadata'])

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # If metadata exists - import it
        inference_metadata = None
        if os.path.exists((filename+'metadata.json')):
            #  Import metadata if it exists
            with open((filename+'metadata.json')) as json_file:
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
        with open((filename+'metadata.json'), 'w') as outfile:
            json.dump(self.inference_metadata, outfile)
        if prints: print(f"File exported to {(filename+'metadata.txt')}")


    def export_plots(self,figs,filename,plot_type,prints:bool=False):

        # Make sure figs is not empty
        if not hasattr(figs,'__len__') or len(figs) < 1 or not all([bool(v) for v in figs]):
            raise ValueError(f'No figures found in {figs}')

        # Loop through each plot and export it
        for i,f in enumerate(figs):
            # Get parameters in string format separated by _
            param_names = "_".join([utils.remove_characters(str(p),latex_characters) for p in figs[i]['parameters']])

            # Export plot to file
            figs[i]['figure'].savefig((filename+f'{plot_type}_{param_names}.png'),dpi=300)
            # Close plot
            plt.close(figs[i]['figure'])

            if prints: print(f"File exported to {(filename+f'{plot_type}_{param_names}.png')}")



    def export_univariate_prior_plots(self,experiment:str='',show_plot:bool=False,prints:bool=False,show_title:bool=True):

        # Get prior plots
        fig = self.generate_univariate_prior_plots(show_plot=show_plot,prints=prints,show_title=show_title)


        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(fig,filename,'prior')

    def export_mcmc_mixing_plots(self,experiment:str='',show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Get subplots
        figs = self.generate_mcmc_mixing_plots(show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"mixing")

    def export_thermodynamic_integration_mcmc_mixing_plots(self,experiment:str='',show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Get subplots
        figs = self.generate_thermodynamic_integration_mcmc_mixing_plots(show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,'thermodynamic_integration',inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"mixing")

    def export_mcmc_parameter_posterior_plots(self,experiment:str='',num_stds:int=2,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Get subplots
        figs = self.generate_mcmc_parameter_posterior_plots(num_stds,show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"parameter_posterior")

    def export_thermodynamic_integration_mcmc_parameter_posterior_plots(self,experiment:str='',num_stds:int=2,show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Get subplots
        figs = self.generate_thermodynamic_integration_mcmc_parameter_posterior_plots(num_stds,show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,'thermodynamic_integration',inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"parameter_posterior")

    def export_mcmc_acf_plots(self,experiment:str='',show_plot:bool=False,show_title:bool=True):

        # Get subplots
        figs = self.generate_mcmc_acf_plots(show_plot=show_plot,show_title=show_title)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"acf")


    def export_vanilla_mcmc_space_exploration_plots(self,experiment:str='',show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Generate plots
        figs = self.generate_vanilla_mcmc_space_exploration_plots(show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"space_exploration")

    def export_thermodynamic_integration_mcmc_space_exploration_plots(self,experiment:str='',show_plot:bool=False,show_title:bool=True,show_sim_param:bool=False):

        # Generate plots
        figs = self.generate_thermodynamic_integration_mcmc_space_exploration_plots(show_plot=show_plot,show_title=show_title,show_sim_param=show_sim_param)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],'thermodynamic_integration',dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,'thermodynamic_integration',inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"space_exploration")


    def export_mcmc_posterior_predictive_plot(self,fundamental_diagram,experiment:str='',num_stds:int=2,show_plot:bool=False,show_title:bool=True):

        # Generate plots
        figs = self.generate_posterior_predictive_plot(fundamental_diagram,num_stds=num_stds,show_plot=show_plot,show_title=show_title)

        if experiment == '':
            # Get inference filename
            filename = utils.prepare_output_inference_filename(self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)
        else:
            # Get experiment filename
            filename = utils.prepare_output_experiment_inference_filename(experiment,inference_id=self.inference_metadata['id'],dataset=self.inference_metadata['data_id'],method=self.method)

        # Export them
        self.export_plots(figs,filename,"posterior_predictive")
