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

# Set flag for testing convergence
convergence_diagnostic = False # True False
# Set flag for exporting prior distribution plots
export_priors = False # True False

# Instantiate objects
inf_model = utils.instantiate_inference_method(inference_id)
fd = utils.instantiate_fundamental_diagram(data_id=inf_model.inference_metadata['data_id'],model=inf_model.inference_metadata['fundamental_diagram'])

seed = str(inf_model.inference_metadata['inference']['vanilla_mcmc']['seed'])
if seed.lower() == "none": seed = None
else: seed = int(seed)

print("Inference id:",inference_id)
print("Data id:",inf_model.inference_metadata['data_id'])
print('seed',seed)
# print(inf_model.inference_metadata['inference']['vanilla_mcmc']['transition_kernel']['proposal_stds'])

# Populate them with data
fd.populate()
inf_model.populate(fd)

# Compute MLE estimate
inf_model.compute_maximum_a_posteriori_estimate(prints=True)

# plt.scatter(fd.rho,np.exp(fd.log_q))
# plt.scatter(fd.rho,fd.log_q)
# plt.plot(fd.rho,fd.simulate(inf_model.transform_parameters(inf_model.map_params,True)),color='red')
# plt.show()


sys.exit()
# print()
# print(inf_model.temperature_schedule[9])
# print(inf_model.temperature_schedule[10])
# print(inf_model.temperature_schedule[11])
# plt.scatter(range(1,len(inf_model.temperature_schedule)+1),inf_model.temperature_schedule)
# plt.plot(range(1,len(inf_model.temperature_schedule)+1),inf_model.temperature_schedule)
# plt.show()
# sys.exit(1)

if export_priors:
    # Plot univariate prior distributions
    inf_model.export_univariate_prior_plots()

# sys.exit(1)

if convergence_diagnostic:
    # Run Vanilla MCMC in parallel and get convergence diagnostic
    vanilla_thetas,vanilla_acceptances = inf_model.run_parallel_mcmc(n=3,type='vanilla_mcmc',prints=True)
    inf_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,prints=True)

    sys.exit(1)

# Compute posterior harmonic mean marginal likelihood estimator
# inf_model.compute_log_posterior_harmonic_mean_estimator(vanilla_thetas,prints=True)

# sys.exit(1)


# Run MCMC
theta_accepted,acceptance = inf_model.vanilla_mcmc(i=0,seed=seed,prints=True)
# Import MCMC samples
# inf_model.import_vanilla_mcmc_samples()

# Export MCMC samples
# inf_model.export_mcmc_samples()

# Export MCMC data
inf_model.export_mcmc_mixing_plots(show_plot=False,show_sim_param=False)
inf_model.export_vanilla_mcmc_space_exploration_plots(show_plot=False,show_sim_param=False)
# inf_model.export_mcmc_acf_plots(show_plot=True)
# inf_model.export_mcmc_parameter_posterior_plots(num_stds=2,show_plot=True,show_sim_param=True)

# Export posterior predictive
# inf_model.export_posterior_predictive()
# inf_model.export_mcmc_posterior_predictive_plot(fd,num_stds=2,show_plot=True)

# Export metadata
inf_model.export_metadata()
