import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
from inference.mcmc_inference_models import GaussianRandomWalkMetropolisHastings
from distutils.util import strtobool

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define simulation id
data_id = "exponential_fd_simulation_small"
# Define experiment id
inference_id = "grwmh_inference_wide_gamma_priors_sigma2_fixed"

# Print parameters
# print(json.dumps(inference_params,indent=2))
# print(json.dumps(simulation_params,indent=2))

# Instantiate specified Fundamental Diagram
fd = utils.instantiate_fundamental_diagram(data_id)

# Instantiate inference model
inf_model,inference_params =  utils.instantiate_inference_method(data_id,inference_id)

sigma2 = None
simulation_params = None
# Load simulation parameters if the data is a simulation
if strtobool(inference_params['simulation_flag']):
    # Load simulation metadata
    fd.import_simulation_metadata(data_id)
    # Get observation noise
    sigma2 = float(fd.true_parameters[-1])

    print('True parameters')
    for i,pname in enumerate(fd.parameter_names):
        print(f'{pname} = {fd.true_parameters[i]}')

# Import simulation data
fd.import_data(data_id)

# Setup inference model
inf_model.setup(inference_params,fd,sigma2)

# Plot univariate prior distributions
# inf_model.export_univariate_prior_plots(fd)

# Compute log unnormalised posterior
# log_true_posterior,parameters_mesh = inf_model.evaluate_log_unnormalised_posterior(fd)
# Export log unnormalised posterior
# inf_model.export_log_unnormalised_posterior(fd,prints=True)
# Import log unnormalised posterior
# inf_model.import_log_unnormalised_posterior(['alpha','beta'])

# Export log unnormalsed log posterior plot
# inf_model.export_log_unnormalised_posterior_plots(fd,True)

# Run MCMC
# theta_accepted,theta_proposed,acceptance = inf_model.vanilla_mcmc(fd,True,None)
# Run thermodynamic integration MCMC
ti_theta_accepted,ti_acceptance = inf_model.thermodynamic_integration_mcmc(fd,True,None)

# Export MCMC samples
inf_model.export_mcmc_samples()
# Import MCMC samples
# inf_model.import_mcmc_samples(fd)

# Export MCMC data
# inf_model.export_mcmc_parameter_posterior_plots(fd,2,True)
# inf_model.export_mcmc_space_exploration_plots(fd,True)
# inf_model.export_mcmc_mixing_plots(fd,True)
# inf_model.export_mcmc_acf_plots(fd,True)
# Export thermodynamic integration MCMC
inf_model.export_thermodynamic_integration_mcmc_mixing_plots(fd,False)

# Compute Gelman and Rubin statistic
# inf_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(prints=True)
# inf_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(prints=True)

# Compute posterior predictive
# inf_model.evaluate_posterior_predictive_moments()
# inf_model.import_posterior_predictive()

# Export posterior predictive
# inf_model.export_posterior_predictive()
# inf_model.export_mcmc_posterior_predictive_plot(fd,2,True)

# Compute marginal likelihood estimators
# inf_model.compute_log_posterior_harmonic_mean_estimator(prints=True)
inf_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(prints=True)

# Export metadata
inf_model.export_metadata()
