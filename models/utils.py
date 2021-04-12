import os, sys
import math
import toml
import json
import collections.abc
import scipy.stats as ss

from fundamental_diagrams.fundamental_diagram_definitions import *
from inference.mcmc_inference_models import *
from probability_distributions import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]


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

# def mylog(x):
#     return np.log(x)
#
# def myexp(x):
#     return np.exp(x)

def myreciprocal(x):
    return 1/x

def myidentity(x):
    return x

def remove_characters(x,chars):
    x = str(x)
    for c in chars:
        if c in x: x = x.replace(c,"")
    return x
""" ---------------------------------------------------------------------------Instantiators-----------------------------------------------------------------------------"""

def instantiate_fundamental_diagram(data_id,model:str=''):
    # Load simulation parameters
    simulation_params = import_simulation_metadata(data_id)
    # print(json.dumps(simulation_params,indent=2))

    # Get class object
    if model == '':
        FD = map_name_to_fundamental_diagram(simulation_params['fundamental_diagram'])
    else:
        FD = map_name_to_fundamental_diagram(model)

    # Instantiate object
    fd = FD(data_id)

    # Update simulation metadata
    fd.simulation_metadata = simulation_params

    # Get flag of whether data are a simulation or not
    fd.simulation_flag = strtobool(fd.simulation_metadata['simulation_flag']) and (model == fd.simulation_metadata['fundamental_diagram'])

    # Import metadata
    fd.store_simulation_data()

    # Return instance of Fundamental Diagram object child class
    return fd

def instantiate_inference_method(inference_id):
    # Load inference parameters
    inference_params = import_inference_metadata(inference_id)
    # Load simulation parameters
    simulation_params = import_simulation_metadata(inference_params['data_id'])

    # Get class object
    Model = map_name_to_inference_method(inference_params['inference_model'])

    # Instantiate object
    inference_model = Model(inference_id)

    # Update inference and  simulation metadata
    inference_model.inference_metadata = inference_params
    inference_model.simulation_metadata = simulation_params

    # Return instance of MarkovChainMonteCarlo object child class
    return inference_model


""" ---------------------------------------------------------------------------Checking existence of attributes/parameters-----------------------------------------------------------------------------"""

def find_lacking_attributes(_self,necessary_attributes):
    # Initialise lacking attributes array
    lacking_attributes = []
    for attr in necessary_attributes:
        if not hasattr(_self,attr):
            lacking_attributes.append(attr)

    return lacking_attributes

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
        return ExponentialFD
    elif name == 'greenshields':
        return GreenshieldsFD
    else:
        raise Exception(f'No fundamental diagram model found for {name.lower()}')

def map_name_to_inference_method(name):

    if name.lower() == 'grwmh':
        return GaussianRandomWalkMetropolisHastings
    else:
        raise Exception(f'No probability distribution found for {name.lower()}')

def map_name_to_parameter_transformation(priors,num_learning_parameters,lower_bounds:list=[],upper_bounds:list=[]):
    transformations = []
    for i,prior in enumerate(list(priors.keys())[0:num_learning_parameters]):
        if 'log' in priors[prior]['transformation'].lower():
            if np.isinf(lower_bounds[i]) and np.isinf(upper_bounds[i]):
                raise ValueError('Cannot apply log transformation to unconstrained variables in map_name_to_parameter_transformation.')
            elif np.isinf(upper_bounds[i]):
                # print('Lower bound')
                transformations.append([(lambda i: lambda x: np.log(x-lower_bounds[i]) )(i) , (lambda i: lambda x: (np.exp(x)+lower_bounds[i]) )(i) ])

            elif np.isinf(lower_bounds[i]):
                # print('Upper bound')
                transformations.append([(lambda i: lambda x: np.log(x/(upper_bounds[i]-x)) )(i) , (lambda i: lambda x: ((upper_bounds[i]*np.exp(x))/(1+np.exp(x))) )(i) ])
            else:
                # print('Upper and lower bounds')
                transformations.append([(lambda i: lambda x: np.log((x-lower_bounds[i])/(upper_bounds[i]-x)) )(i) , (lambda i: lambda x: ((upper_bounds[i]*np.exp(x)+lower_bounds[i])/(1+np.exp(x))) )(i) ])
        elif '1/' in priors[prior]['transformation'].lower():
            transformations.append([myreciprocal, myreciprocal])
        else:
            transformations.append([myidentity, myidentity])
    return transformations


def map_name_to_multivariate_logpdf(name,iid,**kwargs):

    print_statements = False
    if 'prints' in kwargs:
        if kwargs.get('prints'): print_statements = True

    if 'gaussian' in name.lower() or 'normal' in name.lower():
        if print_statements: print(name.lower())
        if iid: return multivariate_gaussian_iid
        else: return multivariate_gaussian
    else:
        raise Exception(f'No suitable likelihood probability distribution found for {name.lower()}')

def map_name_to_univariate_logpdf(name,**kwargs):

    print_statements = False
    if 'prints' in kwargs:
        if kwargs.get('prints'): print_statements = True

    if 'gaussian' in name.lower() or 'normal' in name.lower():
        if print_statements: print(name.lower())
        return gaussian
    else:
        raise Exception(f'No suitable likelihood probability distribution found for {name.lower()}')

def map_name_to_distribution_sampler(name,**kwargs):

    print_statements = False
    if 'prints' in kwargs:
        if kwargs.get('prints'): print_statements = True

    if name.lower() == 'beta':
        if print_statements: print(name.lower())
        return np.random.beta
    elif name.lower() == 'gamma':
        if print_statements: print(name.lower())
        return np.random.gamma
    elif name.lower() in ['mnormal','mgaussian']:
        if print_statements: print(name.lower())
        return np.random.multivariate_normal
    elif name.lower() in ['normal','gaussian']:
        if print_statements: print(name.lower())
        return np.random.normal
    elif name.lower() == 'lognormal' or name.lower() == 'mlognormal':
        if print_statements: print(name.lower())
        return np.random.lognormal
    else:
        raise Exception(f'No numpy probability distribution found for {name.lower()}')

""" ---------------------------------------------------------------------------Prepare filenames for imports/exports-----------------------------------------------------------------------------"""

def prepare_input_simulation_filename(simulation_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/simulation_parameters',simulation_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Simulation file {input_file} not found.')
    # Return simulation filename
    return input_file

def prepare_input_inference_filename(inference_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/inference_parameters',inference_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Inference file {input_file} not found.')

    # Return simulation filename
    return input_file

def prepare_output_simulation_filename(simulation_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/fundamental_diagram_data',simulation_id+'/')
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder

def prepare_output_inference_filename(inference_id,*args,**kwargs):

    # Define output folder path
    if len(args) > 0: output_folder = os.path.join(root,'data/output/inference_data',kwargs.get('dataset'),kwargs.get('method'),inference_id+'/',*[args[i]+'/' for i in range(len(args))])
    else: output_folder = os.path.join(root,'data/output/inference_data',kwargs.get('dataset'),kwargs.get('method'),inference_id+'/')
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder

def prepare_input_experiment_filename(experiment_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/experiment_parameters',experiment_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Experiment file {input_file} not found.')
    # Return simulation filename
    return input_file

def prepare_output_experiment_inference_filename(experiment_id,*args,**kwargs):
    # Define output folder path
    if len(args) > 0: output_folder = os.path.join(root,'data/output/experiment_data',experiment_id,kwargs.get('dataset'),kwargs.get('inference_id')+'/',*[args[i]+'/' for i in range(len(args))])
    else: output_folder = os.path.join(root,'data/output/experiment_data',experiment_id,kwargs.get('dataset'),kwargs.get('inference_id')+'/')
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder

def prepare_output_experiment_simulation_filename(experiment_id,**kwargs):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/experiment_data',experiment_id,kwargs.get('dataset')+'/')

    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder


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

def import_inference_metadata(inference_id):

    # Get data filename
    metadata_filename = utils.prepare_input_inference_filename(inference_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _inference_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _inference_metadata

def import_experiment_metadata(experiment_id):

    # Get data filename
    metadata_filename = utils.prepare_input_experiment_filename(experiment_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _experiment_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _experiment_metadata
