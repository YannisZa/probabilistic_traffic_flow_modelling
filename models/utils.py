import os, sys
import math
from fundamental_diagrams.fundamental_diagram_definitions import *
from inference.mcmc_inference_models import *
import scipy.stats as ss

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# def multivariate_log_normal

def has_necessary_attributes(necessary_attributes):
    # Initialise lacking attributes array
    lacking_attributes = []
    for attr in necessary_attributes:
        if not hasattr(self,('__'+attr)):
            lacking_attributes.append(attr)
            # raise AttributeError(f'Attribute {attr} not found.')

    # If there is at least one lacking attribute raise error
    if len(lacking_attributes) > 0:
        raise AttributeError(f'Attributes {lacking_attributes} not found.')

def has_necessary_parameters(necessary_parameters,current_params):
    # Initialise lacking parameters array
    lacking_params = []
    # Loop through necessary parameters
    for p in necessary_parameters:
        # If parameter does not exist or is null
        if p not in current_params.keys() or current_params[p] == '' or current_params[p] is None:
            lacking_params.append(p)
        if hasattr(p, "__len__") and len(p) < 1:
            lacking_params.append(p)

    # If there is at least one lacking parameter raise error
    if len(lacking_params) > 0:
        raise Exception(f'Parameters {lacking_params} are missing or mispecified.')


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

def map_name_to_class(name):

    if name == 'exponential':
        return ExponentialFD()
    else:
        raise Exception(f'No fundamental diagram model found for {name.lower()}')

def map_name_to_mcmc_method(name):

    if name.lower() == 'grwmh':
        return GaussianRandomWalkMetropolisHastings()
    else:
        raise Exception(f'No probability distribution found for {name.lower()}')


def map_name_to_scipy_distribution(name):

    if name.lower() == 'beta':
        return ss.beta
    elif name.lower() == 'gamma':
        return ss.gamma
    elif name.lower() == 'mnormal':
        return ss.multivariate_normal
    elif name.lower() == 'normal':
        return ss.normal
    elif name.lower() == 'lognormal':
        return ss.lognorm
    else:
        raise Exception(f'No scipy probability distribution found for {name.lower()}')

def map_name_to_numpy_distribution(name):

    if name.lower() == 'beta':
        return np.random.beta
    elif name.lower() == 'gamma':
        return np.random.gamma
    elif name.lower() == 'mnormal':
        return np.random.multivariate_normal
    elif name.lower() == 'normal':
        return np.random.normal
    elif name.lower() == 'lognormal':
        return np.random.lognormal
    else:
        raise Exception(f'No numpy probability distribution found for {name.lower()}')


def prepare_simulation_filename(simulation_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/fundamental_diagram_data',simulation_id)
    # Create new folder if it doesn't exist
    create_dir(output_folder)
    # Return simulation filename
    return os.path.join(output_folder,simulation_id)


def prepare_inference_experiment_filename(dataset,method,inference_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/inference_experiments',dataset,method,inference_id)
    # Create new folder if it doesn't exist
    create_dir(output_folder)
    # Return simulation filename
    return os.path.join(output_folder,inference_id)
