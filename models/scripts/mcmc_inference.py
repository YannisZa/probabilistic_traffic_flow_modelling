import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
import matplotlib.pyplot as plt
from inference.mcmc_inference_models import MetropolisHastings
from distutils.util import strtobool

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define experiment id
inference_id = str(sys.argv[1])

convergence_diagnostic = False # True False

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


# [r'$v_f$',r'$\rho_c$',r'$\rho_j$',r'$\gamma$',r'$\sigma^2$'])
# param = [1.7212974372955934, 15.016179284568013, 44.74034640205288, 26.065566580251833, 44.70995124361817, 1.0, 0.010395255107354152]
#[1.7212974372955934, 15.016179284568013, 44.74034640205288, 26.065566580251833, 0.010395255107354152]

# [r'$v_f$',r'$\rho_c$',r'$\rho_j$',r'$\gamma$',r'$\alpha$',r'$\beta$',r'$\sigma^2$']
# [1.8912121904146477, 26.999964810756076, 50.526424880359315, 193.8806265765878, 33.061141967186764, 1.502221663285689, 0.011036697717723323]
# tparam = inf_model.transform_parameters(param,False)
# lt = inf_model.evaluate_log_target(tparam)
#
# print('lt',lt)
# print()
#
sys.exit(1)

# Plot univariate prior distributions
# inf_model.export_univariate_prior_plots()

# Run Vanilla MCMC in parallel and get convergence diagnostic
# vanilla_thetas,vanilla_acceptances = inf_model.run_parallel_mcmc(n=3,type='vanilla_mcmc',prints=True)
# inf_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,prints=True)

# Compute posterior harmonic mean marginal likelihood estimator
# inf_model.compute_log_posterior_harmonic_mean_estimator(vanilla_thetas,prints=True)

# sys.exit(1)

if convergence_diagnostic:
    # Run Thermodynamic Integration MCMC in parallel and get convergence diagnostic
    ti_thetas,ti_acceptances = inf_model.run_parallel_mcmc(n=3,type='thermodynamic_integration_mcmc',prints=True)
    r_stat,ti_converged,ti_burnin = inf_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,prints=True)

    # Compute thermodynamic integration marginal likelihood estimators
    # inf_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(ti_thetas,prints=True)

    sys.exit(1)

# Run MCMC
# theta_accepted,acceptance = inf_model.vanilla_mcmc(i=0,seed=2021,prints=True)
# Import MCMC samples
# inf_model.import_vanilla_mcmc_samples()

# Export MCMC samples
# inf_model.export_mcmc_samples()

# Run thermodynamic integration MCMC
ti_theta_accepted,ti_proposed = inf_model.thermodynamic_integration_mcmc(i=0,prints=True,seed=None)
# Import MCMC samples
# inf_model.import_thermodynamic_integration_mcmc_samples()

# Export MCMC samples
# inf_model.export_mcmc_samples()

# Export MCMC data
# inf_model.export_mcmc_mixing_plots(show_plot=True,show_sim_param=True)
# inf_model.export_vanilla_mcmc_space_exploration_plots(show_plot=True,show_sim_param=True)
# inf_model.export_mcmc_acf_plots(show_plot=True)
# inf_model.export_mcmc_parameter_posterior_plots(num_stds=2,show_plot=True,show_sim_param=True)

# Export thermodynamic integration MCMC
inf_model.export_thermodynamic_integration_mcmc_space_exploration_plots(show_plot=False,show_sim_param=True)
inf_model.export_thermodynamic_integration_mcmc_mixing_plots(show_plot=False,show_sim_param=True)
# inf_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(num_stds=2,show_plot=False,show_sim_param=True)

# Compute posterior predictive
# inf_model.evaluate_posterior_predictive_moments(prints=True)
# inf_model.import_posterior_predictive()

# Export posterior predictive
# inf_model.export_posterior_predictive()
# inf_model.export_mcmc_posterior_predictive_plot(fd,num_stds=2,show_plot=True)

# Export metadata
inf_model.export_metadata()
