import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
from inference.mcmc_inference_models import GaussianRandomWalkMetropolisHastings

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define simulation id
data_id = "exponential_fd_simulation"
# Define experiment id
inference_id = "grwmh_inference_wide_gamma_priors_sigma2_fixed"


# Load experiment parameters
inference_params = utils.import_inference_metadata(data_id,inference_id)

# Print parameters
# print(json.dumps(inference_params,indent=2))
# print(json.dumps(simulation_params,indent=2))


# Instantiate specified Fundamental Diagram
fd = utils.map_name_to_class(inference_params['fundamental_diagram'])

sigma2 = None
simulation_params = None
# Load simulation parameters if the data is a simulation
if bool(inference_params['simulation_flag']):
    fd.import_simulation_metadata(data_id)
    # Get observation noise
    sigma2 = float(fd.true_parameters[-1])


print('True parameters')
for i,pname in enumerate(fd.parameter_names):
    print(f'{pname} = {fd.true_parameters[i]}')

# Import simulation data
fd.import_data(data_id)

# Instantiate inference model
inf_model =  utils.map_name_to_mcmc_method(inference_params['inference_model'])
# Setup inference model
inf_model.setup(inference_params,fd,sigma2)

# Plot univariate prior distributions
inf_model.export_univariate_prior_plot(fd,data_id)

# Compute unnormalised log posterior
log_true_posterior,log_likelihood,log_prior,parameters_mesh = inf_model.evaluate_unnormalised_log_posterior(fd)

# # Export log unnormalsed log posterior
inf_model.export_unnormalised_log_posterior_plots(fd,data_id)

# Run MCMC
theta_accepted,theta_proposed,acceptance = inf_model.vanilla_mcmc(False,2021)

# Export data
inf_model.export_samples()
inf_model.export_metadata()
inf_model.export_log_unnormalised_posterior()
