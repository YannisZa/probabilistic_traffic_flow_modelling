import os, sys
import math
import toml
import json
import collections.abc
import scipy.stats as ss

from functools import partial
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
    fd.simulation_flag = strtobool(fd.simulation_metadata['simulation_flag']) and ((model == fd.simulation_metadata['fundamental_diagram']) or (model == ''))

    # Import metadata
    fd.store_simulation_data()

    # Return instance of Fundamental Diagram object child class
    return fd

def instantiate_inference_method(inference_id):

    # Get model name from inference_id
    model_name = inference_id.split('_model',1)[0].split("_",1)[1]

    # Load inference parameters
    inference_params = import_inference_metadata(model=model_name,inference_id=inference_id)
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
    elif name == 'daganzos':
        return DaganzosFD
    elif name == 'delcastillos':
        return DelCastillosFD
    elif name == 'greenbergs':
        return GreenbergsFD
    elif name == 'underwoods':
        return UnderwoodsFD
    elif name == 'northwesterns':
        return NorthwesternsFD
    elif name == 'newells':
        return NewellsFD
    elif name == 'wangs':
        return WangsFD
    elif name == 'smulders':
        return SmuldersFD
    elif name == 'deromphs':
        return DeRomphsFD
    elif name == 'deromphs_continuous':
        return DeRomphsContinuousFD
    else:
        raise Exception(f'No fundamental diagram model found for {name.lower()}')

def map_name_to_inference_method(name):

    if any([x in name.lower() for x in ['mh','metropolis_hastings']]):
        return MetropolisHastings
    else:
        raise Exception(f'No probability distribution found for {name.lower()}')

def map_name_to_parameter_transformation(priors,num_learning_parameters,lower_bounds:list=[],upper_bounds:list=[]):
    """Converts transformation name to transformation operator, inverse transformation operator, transformation operator 1st derivative, transformation operator 2nd derivative

    Parameters
    ----------
    priors : dict
        Dictionary of prior names and hypeparameters
    num_learning_parameters : int
        Number of parameters to be learned
    lower_bounds : list
        Lower boundary of transformed variable
    upper_bounds : list
        Upper boundary of transformed variable

    Returns
    -------
    list of lambda functions and string
        Transformation operator, inverse transformation operator, transformation operator 1st derivative, transformation operator 2nd derivative, transformation name

    """
    transformations = []
    for i,prior in enumerate(list(priors.keys())[0:num_learning_parameters]):
        if 'log' in priors[prior]['transformation'].lower():
            if np.isinf(lower_bounds[i]) and np.isinf(upper_bounds[i]):
                raise ValueError('Cannot apply log transformation to unconstrained variables in map_name_to_parameter_transformation.')
            elif np.isinf(upper_bounds[i]):
                # print('Lower bound')
                transformations.append({"forward":(lambda i: lambda x: np.log(x-lower_bounds[i]) )(i) ,
                                        "backward":(lambda i: lambda x: (np.exp(x)+lower_bounds[i]) )(i),
                                        "jacobian":(lambda i: lambda x: 1/(x-lower_bounds[i]) )(i),
                                        "hessian":(lambda i: lambda x: -1/((x-lower_bounds[i])**2) )(i),
                                        "third_derivative":(lambda i: lambda x: 2/((x-lower_bounds[i])**3) )(i),
                                        "name":'log'})

            elif np.isinf(lower_bounds[i]):
                # print('Upper bound')
                transformations.append({"forward":(lambda i: lambda x: np.log(x/(upper_bounds[i]-x)) )(i) ,
                                        "backward":(lambda i: lambda x: ((upper_bounds[i]*np.exp(x))/(1+np.exp(x))) )(i),
                                        "jacobian":(lambda i: lambda x: -upper_bounds[i]/(x*(x-upper_bounds[i])) )(i),
                                        "hessian":(lambda i: lambda x: (upper_bounds[i]*(2*x-upper_bounds[i]))/(x**2*(x-upper_bounds[i])**2) )(i),
                                        "third_derivative":(lambda i: lambda x: -(2*upper_bounds[i]*(3*x**2 - 3*upper_bounds[i]*x + upper_bounds[i]**2))/(x**3*(x-upper_bounds[i])**3) )(i),
                                        "name":'log'})
            else:
                # print('Upper and lower bounds')
                transformations.append({"forward":(lambda i: lambda x: np.log((x-lower_bounds[i])/(upper_bounds[i]-x)) )(i) ,
                                        "backward":(lambda i: lambda x: ((upper_bounds[i]*np.exp(x)+lower_bounds[i])/(1+np.exp(x))) )(i),
                                        "jacobian":(lambda i: lambda x: -(upper_bounds[i]-lower_bounds[i])/((x-lower_bounds[i])*(x-upper_bounds[i])) )(i),
                                        "hessian":(lambda i: lambda x: ((upper_bounds[i]-lower_bounds[i])*(2*x-upper_bounds[i]-lower_bounds[i]))/((x-lower_bounds[i])**2*(x-upper_bounds[i])**2) )(i),
                                        "third_derivative":(lambda i: lambda x: -(2*(upper_bounds[i]-lower_bounds[i])*( 3*x**2 -3*(upper_bounds[i]+lower_bounds[i])*x + upper_bounds[i]**2 + upper_bounds[i]*lower_bounds[i] + upper_bounds[i]**2 ) ) / ( (x-lower_bounds[i])**3 * (x-upper_bounds[i])**3 ) )(i),
                                        "name":'log'})
        elif '1/' in priors[prior]['transformation'].lower():
            print('Need to compute transformation first and second derivatives')
            transformations.append({"forward":myreciprocal,
                                    "backward":myreciprocal,
                                    "jacobian":"",
                                    "hessian":"",
                                    "third_derivative":"",
                                    "name":'1/'})
        else:
            transformations.append({"forward":myidentity,
                                    "backward":myidentity,
                                    "jacobian":(lambda x: 0),
                                    "hessian":(lambda x: 0),
                                    "third_derivative":(lambda x: 0),
                                    "name":''})
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
        return univariate_gaussian
    else:
        raise Exception(f'No suitable likelihood probability distribution found for {name.lower()}')

def map_name_to_distribution_sampler(name,**kwargs):

    print_statements = False
    if 'prints' in kwargs:
        if kwargs.get('prints'): print_statements = True

    if name.lower() == 'beta':
        if print_statements: print(name.lower())
        return np.random.beta
    elif name.lower() == 'uniform':
        if print_statements: print(name.lower())
        return np.random.uniform
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

def map_operation_symbol_to_binary(symbol,lhs,rhs):

    if symbol == '>':
        return int(lhs>rhs)
    elif symbol == '>=':
        return int(lhs>=rhs)
    elif symbol == '==':
        return int(lhs==rhs)
    elif symbol == '!=':
        return int(lhs!=rhs)
    else:
        raise ValueError(f'Symbol {symbol} was not found. Valid symbols are >,>=,==,!=.')

def map_name_to_variable_or_variable_index(self,variable_name,latex_characters):
    # Get parameter names and strip them off special characters
    param_names = np.array([remove_characters(x,latex_characters) for x in self.parameter_names])
    # print('param_names',param_names)
    # print('variable_name',variable_name)
    # IF x or y are in the variable name
    if variable_name in ['max_x','min_x','min_y','max_y']:
        if variable_name == 'max_x':
            return float(max(self.x))
        elif variable_name == 'max_y':
            return float(max(self.y))
        elif variable_name == 'min_x':
            return float(min(self.x))
        elif variable_name == 'min_y':
            return float(min(self.y))
    elif variable_name.isnumeric():
        return float(variable_name)
    elif variable_name in param_names:
        # print("np.where(param_names==variable_name)[0][0]",np.where(param_names==variable_name)[0][0])
        return int(np.where(param_names==variable_name)[0][0])
    else:
        raise ValueError(f"Variable {variable_name} not found. Available choices are parameter names or",", ".join(['max_x','min_x','min_y','max_y']))

def map_constraint_name_to_function(fundamental_diagram,constraint_name:str='',**kwargs):
    # IF x or y are in the variable name
    if constraint_name.lower() == 'capacity_drop':
        # print('Added DeRomphs capacity drop constraint')
        return fundamental_diagram.capacity_drop_constraint
    elif constraint_name.lower() == 'positive_alpha':
        # print('Added DeRomphs positive alpha constraint')
        return fundamental_diagram.positive_alpha_constraint
    else:
        raise ValueError(f"Constaint {constraint_name} not found.")

""" ---------------------------------------------------------------------------Prepare filenames for imports/exports-----------------------------------------------------------------------------"""

def prepare_input_simulation_filename(simulation_id):
    # Define input filepath
    input_file = os.path.join(root,'data/input/simulation_parameters',simulation_id+('.toml'))
    # Ensure file exists
    if not os.path.exists(input_file):
        raise ValueError(f'Simulation file {input_file} not found.')
    # Return simulation filename
    return input_file

def prepare_input_inference_filename(model,inference_id):
    # Define input filepath
    if '_prior' in inference_id:
        # Get prior specification from inference id
        prior = inference_id.split('_prior')[0].rsplit('_',1)[1] + '_prior'
        # Construct inferece parameter input file
        input_file = os.path.join(root,'data/input/inference_parameters',model,prior,inference_id+('.toml'))
    else:
        input_file = os.path.join(root,'data/input/inference_parameters',model,inference_id+('.toml'))
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
    if len(args) > 0:
        output_folder = os.path.join(root,'data/output/experiment_data',kwargs.get('dataset'),kwargs.get('inference_id')+'/',*[args[i]+'/' for i in range(len(args))])
    else:
        output_folder = os.path.join(root,'data/output/experiment_data',kwargs.get('dataset'),kwargs.get('inference_id')+'/')
    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder

def prepare_output_experiment_simulation_filename(experiment_id,**kwargs):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/experiment_data',kwargs.get('dataset')+'/')

    # Create new folder if it doesn't exist
    ensure_dir(output_folder)
    # Return simulation filename
    return output_folder

def prepare_output_experiment_summary_filename(experiment_id):
    # Define output folder path
    output_folder = os.path.join(root,'data/output/experiment_data/table_summaries/')
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

def import_inference_metadata(model,inference_id):

    # Get data filename
    metadata_filename = prepare_input_inference_filename(model=model,inference_id=inference_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _inference_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _inference_metadata

def import_experiment_metadata(experiment_id):

    # Get data filename
    metadata_filename = prepare_input_experiment_filename(experiment_id)

    # Import simulation metadata
    if os.path.exists(metadata_filename):
        _experiment_metadata  = toml.load(metadata_filename)
    else:
        raise FileNotFoundError(f'File {metadata_filename} not found.')

    return _experiment_metadata
