import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import numpy as np
from utils import map_name_to_class,prepare_simulation_filename,map_name_to_mcmc_method
from inference.mcmc_inference_models import GaussianRandomWalkMetropolisHastings

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define simulation id
data_id = "sample_simulation_id"
# Define experiment id
experiment_id = "sample_experiment_id"


# Define path to simulation parameters
simulation_parameters_filepath = os.path.join(root,f'data/input/simulation_parameters/{data_id}.toml')
# Define path to inference experiment parameters
inference_experiment_parameters_filepath = os.path.join(root,f'data/input/inference_parameters/{experiment_id}.toml')

# Load experiment parameters
if os.path.exists(inference_experiment_parameters_filepath):
    experiment_params  = toml.load(inference_experiment_parameters_filepath)
else:
    raise FileNotFoundError(f'File {inference_experiment_parameters_filepath} not found.')

sigma2 = None
simulation_params = None
# Load simulation parameters if the data is a simulation
if experiment_params['simulation_flag']:
    if os.path.exists(simulation_parameters_filepath):
        simulation_params  = toml.load(simulation_parameters_filepath)
    else:
        raise FileNotFoundError(f'File {simulation_parameters_filepath} not found.')
    # Get observation noise
    sigma2 = float(simulation_params['simulation']['sigma2'])

# Print parameters
# json.dumps(experiment_params,indent=2)
# print(json.dumps(simulation_params,indent=2))

# Instantiate specified Fundamental Diagram
fd = map_name_to_class(experiment_params['fundamental_diagram'])

# Get simulation filename
data_filename = prepare_simulation_filename(data_id)

# Import simulation data
fd.import_data(data_filename)

# Instantiate inference model
inf_model =  map_name_to_mcmc_method(experiment_params['inference_model'])

# Update data and parameters in inference model
inf_model.update_data(fd.rho,fd.q)

# Update model likelihood
inf_model.update_log_likelihood_log_pdf(experiment_params,fd,sigma2)

# print(inf_model.evaluate_log_likelihood([0.6,0.1]))

# Update model priors
inf_model.update_log_prior_log_pdf(experiment_params,fd.parameter_number)

# print(inf_model.evaluate_log_joint_prior([0.6,0.1]))

# Update model transition kernel
inf_model.update_transition_kernel(experiment_params,fd.parameter_number)

# Propose new sample
p_new = inf_model.propose_new_sample(np.array([0.6,0.1]))

# Plot univariate prior distributions
# inf_model.plot_univariate_priors(experiment_params,simulation_params,fd)

# Sample from prior distributions
# samples = inf_model.sample_from_univariate_priors(experiment_params,fd.parameter_number,1)


# Run MCMC
theta_accepted,theta_proposed,acceptance = inf_model.vanilla_mcmc(experiment_params,True)
print(f'Acceptance rate {int(grw_mcmc_acceptance*100)}%')
