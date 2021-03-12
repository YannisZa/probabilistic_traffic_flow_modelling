import os, sys
import math
import toml
import collections.abc
import scipy.stats as ss

from fundamental_diagrams.fundamental_diagram_definitions import *
from inference.mcmc_inference_models import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]

# def multivariate_log_normal

""" ---------------------------------------------------------------------------General purpose-----------------------------------------------------------------------------"""
def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def ensure_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


""" ---------------------------------------------------------------------------Instantiators-----------------------------------------------------------------------------"""

def instantiate_fundamental_diagram(data_id):
    # Load simulation parameters
    simulation_params = import_simulation_metadata(data_id)

    # Return instance of Fundamental Diagram object child class
    return utils.map_name_to_fundamental_diagram(simulation_params['fundamental_diagram'])

def instantiate_inference_method(data_id,inference_id):
    # Load inference parameters
    inference_params = import_inference_metadata(data_id,inference_id)

    # Return instance of MarkovChainMonteCarlo object child class
    return utils.map_name_to_inference_method(inference_params['inference_model']),inference_params


""" ---------------------------------------------------------------------------Checking existence of attributes/parameters-----------------------------------------------------------------------------"""

def find_lacking_attributes(_self,necessary_attributes):
    # Initialise lacking attributes array
    lacking_attributes = []
    for attr in necessary_attributes:
        if not hasattr(_self,attr):
            lacking_attributes.append(attr)

    return lacking_attributes

def has_attributes(_self,necessary_attributes):
    # Find lacking attributes
    lacking_attributes =  find_lacking_attributes(_self,necessary_attributes)

    # If there is at least one lacking attribute raise error
    if len(lacking_attributes) > 0: return False
    else: return True


def validate_attribute_existence(_self,necessary_attributes):
    # Find lacking attributes
    lacking_attributes =  find_lacking_attributes(_self,necessary_attributes)

    # If there is at least one lacking attribute raise error
    if len(lacking_attributes) > 0:
        raise AttributeError(f'Attributes {lacking_attributes} not found.')

def find_lacking_parameters(necessary_parameters,current_params):
    # Initialise lacking parameters array
    lacking_params = []
    # Loop through necessary parameters
    for p in necessary_parameters:
        # If parameter does not exist or is nullprepare_simulation_filename
        if p not in current_params.keys() or current_params[p] == '' or current_params[p] is None:
            lacking_params.append(p)
        if hasattr(p, "__len__") and len(p) < 1:
            lacking_params.append(p)

    return lacking_params

def has_parameters(necessary_parameters,current_params):

    # Find lacking parameters
    lacking_params =  find_lacking_parameters(necessary_parameters,current_params)

    # If there is at least one lacking parameter raise error
    if len(lacking_params) > 0: return False
    else: return True


def validate_parameter_existence(necessary_parameters,current_params):
    # Find lacking parameters
    lacking_params =  find_lacking_parameters(necessary_parameters,current_params)

    # If there is at least one lacking parameter raise error
    if len(lacking_params) > 0:
        raise Exception(f'Parameters {lacking_params} are missing or mispecified.')



""" ---------------------------------------------------------------------------String to function maps-----------------------------------------------------------------------------"""

def map_name_to_fundamental_diagram(name):

    if name == 'exponential':
        return ExponentialFD()
    else:
        raise Exception(f'No fundamental diagram model found for {name.lower()}')

def map_name_to_inference_method(name):

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
    elif name.lower() == 'lognormal' or name.lower() == 'mlognormal':
        return np.random.lognormal
    else:
        raise Exception(f'No numpy probability distribution found for {name.lower()}')

""" ---------------------------------------------------------------------------Prepare filenames for imports/exports-----------------------------------------------------------------------------"""


def prepare_output_simulation_filename(simulation_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/fundamental_diagram_data',simulation_id)
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return os.path.join(output_folder,simulation_id)

def prepare_output_inference_filename(dataset,method,inference_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/inference_experiments',dataset,method,inference_id)
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return os.path.join(output_folder,inference_id)

def prepare_input_simulation_filename(simulation_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/simulation_parameters',simulation_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Simulation file {input_file} not found.')
    # Return simulation filename
    return input_file

def prepare_input_inference_filename(data_id,inference_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/inference_parameters',inference_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Simulation file {input_file} not found.')

    # Return simulation filename
    return input_file

""" ---------------------------------------------------------------------------Import metadata from file-----------------------------------------------------------------------------"""

def import_simulation_metadata(data_id):

    # Get data filename
    metadata_filename = utils.prepare_input_simulation_filename(data_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _simulation_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _simulation_metadata

def import_inference_metadata(data_id,inference_id):

    # Get data filename
    metadata_filename = utils.prepare_input_inference_filename(data_id,inference_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _inference_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _inference_metadata
