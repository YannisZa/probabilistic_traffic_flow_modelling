import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
import matplotlib.pyplot as plt
from inference.mcmc_inference_models import GaussianRandomWalkMetropolisHastings
from distutils.util import strtobool

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define simulation id
# data_id = "exponential_fd_simulation_smaller"
#"exponential_fd_simulation_small_medium_noise"
#"exponential_fd_simulation_small"
# Define experiment id
inference_id = "grwmh_inference_wide_gamma_priors_sigma2_fixed_small"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed_smaller_more_data"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed_small"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed"

# Print parameters
# print(json.dumps(inference_params,indent=2))
# print(json.dumps(simulation_params,indent=2))

# Instantiate objects
inf_model = utils.instantiate_inference_method(inference_id)
fd = utils.instantiate_fundamental_diagram(inf_model.inference_metadata['data_id'])

print("Inference id:",inference_id)
print("Data id:",inf_model.inference_metadata['data_id'])

# Populate them with data
fd.populate()
inf_model.populate(fd)

# Compute MLE estimate
inf_model.compute_mle_estimate(fd,prints=True)

# Plot univariate prior distributions
# inf_model.export_univariate_prior_plots(fd)

# Compute log unnormalised posterior
# log_true_posterior,parameters_mesh = inf_model.evaluate_log_unnormalised_posterior(fd)
# Import log unnormalised posterior
inf_model.import_log_unnormalised_posterior(['alpha','beta'])

# Export log unnormalised posterior
# inf_model.export_log_unnormalised_posterior(fd,prints=True)
# Export log unnormalsed log posterior plot
# inf_model.export_log_unnormalised_posterior_plots(fd,True)

# Run MCMC
# theta_accepted,theta_proposed,acceptance = inf_model.vanilla_mcmc(fd,True,None)
# Run thermodynamic integration MCMC
ti_theta_accepted,ti_accepted,ti_proposed = inf_model.thermodynamic_integration_mcmc(fd,True,None)

# Export MCMC samples
# inf_model.export_mcmc_samples()
# Import MCMC samples
# inf_model.import_mcmc_samples(fd)

# Export MCMC data
# inf_model.export_mcmc_parameter_posterior_plots(fd,2,True)
# inf_model.export_mcmc_space_exploration_plots(fd,True)
# inf_model.export_mcmc_mixing_plots(fd,True)
# inf_model.export_mcmc_acf_plots(fd,True)
# Export thermodynamic integration MCMC
inf_model.export_thermodynamic_integration_mcmc_mixing_plots(fd,False)
inf_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(fd,2,False)
inf_model.export_thermodynamic_integration_mcmc_space_exploration_plots(fd,False)

# Compute posterior predictive
# inf_model.evaluate_posterior_predictive_moments()
# inf_model.import_posterior_predictive()

# Export posterior predictive
# inf_model.export_posterior_predictive()
# inf_model.export_mcmc_posterior_predictive_plot(fd,2,True)


# Compute Gelman and Rubin statistic
# inf_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(prints=True)
# inf_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(prints=True)


# Compute marginal likelihood estimators
# inf_model.compute_log_posterior_harmonic_mean_estimator(prints=True)
# inf_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(prints=True)

# Export metadata
inf_model.export_metadata()
