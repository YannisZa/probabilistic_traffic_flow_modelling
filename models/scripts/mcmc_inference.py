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
inference_id = str(sys.argv[1])
#"grwmh_inference_wide_gamma_priors_sigma2_fixed_small"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed_smaller_more_data"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed_small"
#"grwmh_inference_wide_gamma_priors_sigma2_fixed"

# Print parameters
# print(json.dumps(inference_params,indent=2))
# print(json.dumps(simulation_params,indent=2))

# Instantiate objects
inf_model = utils.instantiate_inference_method(inference_id)
fd = utils.instantiate_fundamental_diagram(data_id=inf_model.inference_metadata['data_id'],model=inf_model.inference_metadata['fundamental_diagram'])

print("Inference id:",inference_id)
print("Data id:",inf_model.inference_metadata['data_id'])

# Populate them with data
fd.populate()
inf_model.populate(fd)

# Compute MLE estimate
inf_model.compute_maximum_a_posteriori_estimate(prints=True)

# sys.exit(1)

# Plot univariate prior distributions
inf_model.export_univariate_prior_plots()

# Compute log unnormalised posterior
# log_true_posterior,parameters_mesh = inf_model.evaluate_log_unnormalised_posterior()
# Import log unnormalised posterior
# inf_model.import_log_unnormalised_posterior(['alpha','beta'])

# Export log unnormalised posterior
# inf_model.export_log_unnormalised_posterior(prints=True)
# Export log unnormalsed log posterior plot
# inf_model.export_log_unnormalised_posterior_plots(show_plot=True)

# Run Vanilla MCMC in parallel and get convergence diagnostic
vanilla_thetas = inf_model.run_parallel_mcmc(type='vanilla_mcmc',prints=True)
inf_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,prints=True)

# Run Thermodynamic Integration MCMC in parallel and get convergence diagnostic
ti_thetas = inf_model.run_parallel_mcmc(type='thermodynamic_integration_mcmc',prints=True)
inf_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,prints=True)

# sys.exit(1)

# Run MCMC
# theta_accepted,acceptance = inf_model.vanilla_mcmc(i=0,seed=2021,prints=True)
# Import MCMC samples
# inf_model.import_vanilla_mcmc_samples()

# Run thermodynamic integration MCMC
# ti_theta_accepted,ti_proposed = inf_model.thermodynamic_integration_mcmc(i=0,prints=True,seed=None)
# Import MCMC samples
# inf_model.import_thermodynamic_integration_mcmc_samples()

# Export MCMC samples
# inf_model.export_mcmc_samples()

# Export MCMC data
# inf_model.export_mcmc_parameter_posterior_plots(num_stds=2,show_plot=True,show_sim_param=True)
# inf_model.export_vanilla_mcmc_space_exploration_plots(show_plot=True,show_sim_param=True)
# inf_model.export_mcmc_mixing_plots(show_plot=True,show_sim_param=True)
# inf_model.export_mcmc_acf_plots(show_plot=True)
# Export thermodynamic integration MCMC
# inf_model.export_thermodynamic_integration_mcmc_mixing_plots(show_plot=False,show_sim_param=True)
# inf_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(num_stds=2,show_plot=False,show_sim_param=True)
# inf_model.export_thermodynamic_integration_mcmc_space_exploration_plots(show_plot=False,show_sim_param=True)

# Compute posterior predictive
# inf_model.evaluate_posterior_predictive_moments(prints=True)
# inf_model.import_posterior_predictive()

# Export posterior predictive
# inf_model.export_posterior_predictive()
# inf_model.export_mcmc_posterior_predictive_plot(fd,num_stds=2,show_plot=True)


# Compute marginal likelihood estimators
# inf_model.compute_log_posterior_harmonic_mean_estimator(prints=True)
# inf_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(prints=True)

# Export metadata
inf_model.export_metadata()
