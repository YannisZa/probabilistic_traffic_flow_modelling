import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import re
import time
import toml
import json
import copy
import utils
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as so

from tqdm import tqdm
from distutils.util import strtobool
from fundamental_diagrams import FundamentalDiagram
from inference import MarkovChainMonteCarlo

matplotlib.rc('font', **{'size': 18})

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]

# Define min and max acceptance rate that all experiments should have
min_acceptance = 10.0#33.0#9.0#35.0
max_acceptance = 60.0#51.0#51.0

# Function that substracts strings expressed with uncertainties
def subtract_lmls(a,b):
    if a in ['nan','tuning_problem'] or b in ['nan','tuning_problem']:
        return 'nan'
    # Get a mean and uncertainty
    a_mean = float(a.split('+/-')[0])
    a_uncertainty = float(a.split('+/-')[1])
    # Get b mean and uncertainty
    b_mean = float(b.split('+/-')[0])
    b_uncertainty = float(b.split('+/-')[1])
    # Subtract the two quantities and add up errors
    result_mean = np.round(a_mean - b_mean,2)
    result_uncertainty = np.round(a_uncertainty + b_uncertainty,2)
    return str(result_mean)+"+/-"+str(result_uncertainty)


class Experiment(object):

    def __init__(self,experiment_id):
        # Import metadata / experiment parameters
        self.experiment_metadata = utils.import_experiment_metadata(experiment_id)
        self.experiment_id = experiment_id

    def valid_input(self):

        # Flag for proceeding with experiments
        proceed = True

        # Inference ids and simulation ids must be equal in number
        if len(list(self.experiment_metadata['inference_ids'])) != len(list(self.experiment_metadata['data_ids'])):
            proceed = False
            print(f"Inference and data ids are not equal in number {len(list(self.experiment_metadata['inference_ids']))} != {len(list(self.experiment_metadata['data_ids']))}")

        return proceed

    def run_sequentially(self):

        # Get starting time
        start = time.time()

        if strtobool(self.experiment_metadata['routines']['generate_data']):
            for data_id in set(list(self.experiment_metadata['data_ids'])):
                self.generate_data(data_id)

            # Wait for 10 seconds for files to be exported properly
            time.sleep(10)

        if 'tune_inference' in self.experiment_metadata['routines'].keys() and strtobool(self.experiment_metadata['routines']['tune_inference']):
            for inference_id in list(self.experiment_metadata['inference_ids']):
                self.tune_inference(inference_id)


        if strtobool(self.experiment_metadata['routines']['run_inference']):
            for inference_id in list(self.experiment_metadata['inference_ids']):
                self.run_inference(inference_id)

        # Update relevant flag
        compile_marginal_likelihood_matrix = False
        if "compile_marginal_likelihood_matrix" in self.experiment_metadata['routines']:
            compile_marginal_likelihood_matrix = strtobool(self.experiment_metadata['routines']['compile_marginal_likelihood_matrix'])
        # Compute ML and R2 tables if instructed
        if compile_marginal_likelihood_matrix:
            self.compile_marginal_likelihood_matrix(list(self.experiment_metadata['data_ids']),
                                                    list(self.experiment_metadata['inference_ids']),
                                                    experiment_id=self.experiment_id,
                                                    prints=strtobool(self.experiment_metadata['experiment_summary']['print']),
                                                    export=strtobool(self.experiment_metadata['experiment_summary']['export']))
        # Update relevant flag
        compile_r2_matrix = False
        if "compile_r2_matrix" in self.experiment_metadata['routines']:
            compile_r2_matrix = strtobool(self.experiment_metadata['routines']['compile_r2_matrix'])
        if compile_r2_matrix:
            self.compile_r2_matrix(list(self.experiment_metadata['data_ids']),
                                    list(self.experiment_metadata['inference_ids']),
                                    experiment_id=self.experiment_id,
                                    prints=strtobool(self.experiment_metadata['experiment_summary']['print']),
                                    export=strtobool(self.experiment_metadata['experiment_summary']['export']))

        # Update relevant flag
        compile_sensitivity_analysis_marginal_likelihood_matrix = False
        if "compile_sensitivity_analysis_marginal_likelihood_matrix" in self.experiment_metadata['routines']:
            compile_sensitivity_analysis_marginal_likelihood_matrix = strtobool(self.experiment_metadata['routines']['compile_sensitivity_analysis_marginal_likelihood_matrix'])
        # Compute sensitivity ML table if instructed
        if compile_sensitivity_analysis_marginal_likelihood_matrix:
            self.compile_sensitivity_analysis_marginal_likelihood_matrix(list(self.experiment_metadata['data_ids']),
                                    list(self.experiment_metadata['inference_ids']),
                                    experiment_id=self.experiment_id,
                                    prints=strtobool(self.experiment_metadata['experiment_summary']['print']),
                                    export=strtobool(self.experiment_metadata['experiment_summary']['export']))

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        experiment_id = str(self.experiment_metadata['id'])
        print(f"Experiment "+self.experiment_metadata['id']+" finished in {:0>2}:{:0>2}:{:05.2f} hours...".format(int(hours),int(minutes),seconds))

    def generate_data(self,data_id):

        # Instantiate specified Fundamental Diagram
        fd = utils.instantiate_fundamental_diagram(data_id)

        # Simulate data with noise
        fd.simulate_with_noise(fd.true_parameters)

        # Export plot and data
        fd.export_simulation_plot(experiment_id=str(self.experiment_metadata['id']),
                                plot_log=strtobool(self.experiment_metadata['data_simulation']['plot_log_data']),
                                show_plot=strtobool(self.experiment_metadata['data_simulation']['show_plot']),
                                prints=strtobool(self.experiment_metadata['data_simulation']['print']))

        fd.export_data(experiment_id=str(self.experiment_metadata['id']),
                    prints=strtobool(self.experiment_metadata['data_simulation']['print']))


    def run_inference(self,inference_id):

        print(f'------------------------------------Inference id: {inference_id} ------------------------------------')

        # Get starting time
        start = time.time()

        # Ensure you provide valid inputs
        if not self.valid_input(): raise ValueError(f"Cannot proceed with experiment {self.experiment_metadata['id']}")

        # Instantiate objects
        inference_model = utils.instantiate_inference_method(inference_id)
        fd = utils.instantiate_fundamental_diagram(data_id=inference_model.inference_metadata['data_id'],model=inference_model.inference_metadata['fundamental_diagram'])

        # Populate them with data
        fd.populate(experiment_id=str(self.experiment_metadata['id']))
        inference_model.populate(fd)

        # Compute MLE estimate
        if strtobool(self.experiment_metadata['mle']['compute']):
            inference_model.compute_maximum_a_posteriori_estimate(prints=strtobool(self.experiment_metadata['mle']['print']))

        # Plot univariate prior distributions
        if strtobool(self.experiment_metadata['priors']['export']):
            # print('Export priors')
            inference_model.export_univariate_prior_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))
        elif strtobool(self.experiment_metadata['priors']['show_plot']):
            inference_model.generate_univariate_prior_plots(show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))

        # Compute convergence criterion for Vanilla MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['compute']):
            vanilla_thetas,vanilla_acceptances = inference_model.run_parallel_mcmc(type='vanilla_mcmc',
                                                                prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,
                                                                            prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
            print("\n")
        # Compute convergence criterion for Thermodynamic Integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['compute']):
            ti_thetas,ti_acceptances = inference_model.run_parallel_mcmc(type='thermodynamic_integration_mcmc',
                                                    prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,
                                                                                        prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))
            print("\n")

        # Marginal likelihood estimators
        # Compute Vanilla MCMC marginal likelihood estimator
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['compute']):
                print('Compute posterior harmonic mean marginal likelihood estimator')
                inference_model.compute_log_posterior_harmonic_mean_estimator(vanilla_thetas,prints=strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['print']))
                print("\n")

        # Compute thermodynamic integration MCMC marginal likelihood estimator
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['compute']):
                print('Compute thermodynamic integration marginal likelihood estimator')
                inference_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(ti_thetas,prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['print']))
                print("\n")

        # Run Vanilla MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            # Import Vanilla MCMC chain
            print('Import Vanilla MCMC samples')
            inference_model.import_vanilla_mcmc_samples(experiment=self.experiment_metadata['id'])
            print("\n")
        elif not strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            print('Run Vanilla MCMC')
            theta_accepted,acceptance = inference_model.vanilla_mcmc(i = 0,
                                                                    seed = int(inference_model.inference_metadata['inference']['vanilla_mcmc']['seed']),
                                                                    prints = strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['print']))
            print("\n")

        # Run thermodynamic integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            # Import Thermodynamic Integration MCMC chain
            print('Import Thermodynamic Integration MCMC samples')
            inference_model.import_thermodynamic_integration_mcmc_samples(experiment=self.experiment_metadata['id'])
            print("\n")
        elif not strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            print('Run Thermodynamic Integration MCMC')
            ti_theta_accepted,ti_acceptance = inference_model.thermodynamic_integration_mcmc(i=0,
                                                                                    seed = int(inference_model.inference_metadata['inference']['thermodynamic_integration_mcmc']['seed']),
                                                                                    prints = strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['print']))
            print("\n")

        # Export MCMC chains
        if (strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']) and not strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']))\
            or (strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute'])):
            print('Export MCMC samples')
            inference_model.export_mcmc_samples(experiment=str(self.experiment_metadata['id']))
            print("\n")


        # Export vanilla MCMC plots
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']):
                print('Export Vanilla MCMC plots')
                inference_model.export_mcmc_parameter_posterior_plots(experiment=str(self.experiment_metadata['id']),
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                inference_model.export_vanilla_mcmc_space_exploration_plots(experiment=str(self.experiment_metadata['id']),
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                inference_model.export_mcmc_mixing_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                        show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                inference_model.export_mcmc_acf_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                print("\n")
            elif strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']):
                    _ = inference_model.generate_mcmc_parameter_posterior_plots(num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                    _ = inference_model.generate_vanilla_mcmc_space_exploration_plots(show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                    _ = inference_model.generate_mcmc_mixing_plots(show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']),
                                                                show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                    _ = inference_model.generate_mcmc_acf_plots(show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
        # Export thermodynamic integration MCMC plots
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
                print('Export thermodynamic integration MCMC plots')
                inference_model.export_thermodynamic_integration_mcmc_mixing_plots(experiment=str(self.experiment_metadata['id']),
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                # REMOVE THIS BEFORE FORMAL EXPERIMENTS
                # inference_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(experiment=str(self.experiment_metadata['id']),
                #                                                                     num_stds=2,
                #                                                                     show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                #                                                                     show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                #                                                                     show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                inference_model.export_thermodynamic_integration_mcmc_space_exploration_plots(experiment=str(self.experiment_metadata['id']),
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                print("\n")
            elif strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']):
                _ = inference_model.generate_thermodynamic_integration_mcmc_mixing_plots(show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_parameter_posterior_plots(num_stds=2,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_space_exploration_plots(show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']),
                                                                                    show_sim_param=strtobool(self.experiment_metadata['data_simulation']['show_sim_param']))

        # Import/Compute posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']):
            print('Import posterior predictive')
            inference_model.import_posterior_predictive(experiment=str(self.experiment_metadata['id']))
            print("\n")
        elif not strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']):
            print('Compute posterior predictive')
            inference_model.evaluate_posterior_predictive_moments(prints=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['print']))
            print("\n")

        # Compute R2 based on posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']) and strtobool(self.experiment_metadata['vanilla_mcmc']['R2']['compute']):
            print('Compute posterior predictive R2')
            inference_model.evaluate_posterior_predictive_r_squared(prints=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['print']))
            print("\n")

        # Expore/Store posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']):
            if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['export']):
                print('Export posterior predictive')
                inference_model.export_posterior_predictive(experiment=str(self.experiment_metadata['id']))
                inference_model.export_mcmc_posterior_predictive_plot(fd,
                                                                        experiment=str(self.experiment_metadata['id']),
                                                                        num_stds=2,
                                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_plot']),
                                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_title']))
                print("\n")
            elif strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_plot']):
                _ = inference_model.generate_posterior_predictive_plot(fd,
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_title']))

        # Export metadata
        inference_model.export_metadata(experiment=str(self.experiment_metadata['id']))

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        experiment_id = str(self.experiment_metadata['id'])
        print(f"Inference "+inference_id+" finished in {:0>2}:{:0>2}:{:05.2f} hours...".format(int(hours),int(minutes),seconds))
        print("\n")
        print("\n")

    def compile_marginal_likelihood_matrix(self,data_ids,inference_ids,experiment_id,inference_method='grwmh',prints:bool=False,export:bool=False):

        # Initialise list of rows of log marginal likelihoods
        # vanilla_mcmc_lmls = []
        ti_mcmc_lmls = []
        ti_mcmc_mean_lmls = []

        # Get data FDs
        data_fds = np.array([d.split('_fd')[0] for d in data_ids])
        # Get unique data FDs
        data_fds = np.unique(data_fds)
        # Get model FDs
        model_fds = [i.split('_model')[0].split('_')[1] for i in inference_ids]

        # Loop through data ids
        for data_id in np.unique(data_ids):#tqdm(data_ids):
            # Get data FD
            data_fd = data_id.split('_fd',1)[0]
            # Import data simulation parameter
            data_params = utils.import_simulation_metadata(data_id)

            # Get flag for whether data is simulation
            simulation_data = bool(strtobool(data_params['simulation_flag']))
            experiment_type = "n" + str(data_params['id'].split("_n")[1])

            # Create inference ids
            if simulation_data:
                inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_sim_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]
            else:
                inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]
            # Loop through constructed inference ids
            for i,inference_id in tqdm(enumerate(inference_ids)):

                # Get inference model name
                inference_fd = model_fds[i]
                inference_id = inference_ids[i]

                # print('inference_id',inference_id)
                # print('data_fd',data_fd)
                # print('inference_fd',inference_fd)
                # print('\n')
                # sys.exit(1)

                # Define experiment metadata filename
                metadata_filename = utils.prepare_output_experiment_inference_filename(experiment_id=self.experiment_metadata['id'],inference_id=inference_id,dataset=data_id,method=inference_method)

                # Make sure file exists
                if not os.path.exists((metadata_filename+'metadata.json')):
                    if prints: print(f"Metadata file {metadata_filename}metadata.json not found")
                    # vanilla_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                    ti_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                    ti_mcmc_mean_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                    continue


                #  Import metadata where acceptance is part of metadata
                with open((metadata_filename+'metadata.json')) as json_file:
                    inference_metadata = json.load(json_file)

                # if 'vanilla_mcmc' in inference_metadata['results'].keys():
                #     # Get convergence flag
                #     # Check that Gelman and Rubin-inferred burnin is lower than used burnin
                #     # Check that acceptance rate is between 40% and 50%
                #     vanilla_mcmc_converged = all([bool(inference_metadata['results']['vanilla_mcmc']['converged']),
                #                                 int(inference_metadata['results']['vanilla_mcmc']['burnin']) <= int(inference_metadata['inference']['vanilla_mcmc']['burnin']),
                #                                 float(inference_metadata['results']['vanilla_mcmc']['acceptance_rate']) >= min_acceptance,
                #                                 float(inference_metadata['results']['vanilla_mcmc']['acceptance_rate']) <= max_acceptance])
                #
                #
                #     if not vanilla_mcmc_converged:
                #         print('Vanilla mcmc data fd:',data_fd,'inference fd:',inference_fd)
                #         print('acceptance rate:',json.dumps(inference_metadata['results']['vanilla_mcmc']['acceptance_rate'],indent=2))
                #         print('burnin:',json.dumps(inference_metadata['results']['vanilla_mcmc']['burnin'],indent=2))
                #         print('\n')
                #
                #     # Add log marginal likelihood mean and var to records only if convergence was achieved
                #     if vanilla_mcmc_converged:
                #         # Get log marginal likelihood mean variance for vanilla MCMC
                #         vanilla_lml_mean = np.round(float(inference_metadata['results']['vanilla_mcmc']['log_marginal_likelihoods_mean']),2)
                #         vanilla_lml_var = np.round(float(inference_metadata['results']['vanilla_mcmc']['log_marginal_likelihoods_var']),2)
                #         # Compute them into a string
                #         vanilla_mcmc_lml_entry = str(vanilla_lml_mean)+' +/- '+str(vanilla_lml_var)
                #     else:
                #         vanilla_mcmc_lml_entry = 'tuning_problem'
                # else:
                #     vanilla_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])

                # DITTO for thermodynamic integration
                if 'thermodynamic_integration_mcmc' in inference_metadata['results'].keys():
                    # Get convergence flag and check that Gelman and Rubin-inferred burnin is lower than used burnin
                    ti_mcmc_converged = all([bool(inference_metadata['results']['thermodynamic_integration_mcmc']['converged']),
                                            int(inference_metadata['results']['thermodynamic_integration_mcmc']['burnin']) <= int(inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin']),
                                            float(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate']) >= min_acceptance,
                                            float(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate']) <= max_acceptance])

                    if not ti_mcmc_converged:
                        print('Thermodynamic Integration mcmc data fd:',data_fd,'inference fd:',inference_fd)
                        print('acceptance rate:',json.dumps(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate'],indent=2))
                        print('burnin:',json.dumps(inference_metadata['results']['thermodynamic_integration_mcmc']['burnin'],indent=2))
                        print('\n')
                    # Add log marginal likelihood mean and var to records only if convergence was achieved
                    if ti_mcmc_converged:
                        # Get log marginal likelihood mean variance for thermodynamic integration MCMC
                        ti_lml_mean = np.round(float(inference_metadata['results']['thermodynamic_integration_mcmc']['log_marginal_likelihoods_mean']),2)
                        ti_lml_var = np.round(float(inference_metadata['results']['thermodynamic_integration_mcmc']['log_marginal_likelihoods_var']),2)
                        # Compute them into a string
                        ti_mcmc_lml_entry = str(ti_lml_mean)+' +/- '+str(ti_lml_var)
                        ti_mcmc_mean_lmls_entry = ti_lml_mean
                    else:
                        ti_mcmc_lml_entry = 'tuning_problem'
                        ti_mcmc_mean_lmls_entry = 'nan'

                    # Append entry to results
                    # vanilla_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),vanilla_mcmc_lml_entry])
                    ti_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),ti_mcmc_lml_entry])
                    ti_mcmc_mean_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),ti_mcmc_mean_lmls_entry])
                else:
                    # print('data_id',data_id)
                    # print('inference_id',inference_id)
                    # print('\n')
                    ti_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                    ti_mcmc_mean_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])


        # Convert to np array
        # vanilla_mcmc_lmls = np.array(vanilla_mcmc_lmls)
        ti_mcmc_lmls = np.array(ti_mcmc_lmls)
        ti_mcmc_mean_lmls = np.array(ti_mcmc_mean_lmls)

        # # Get list of unique data models
        # data_fds = np.unique(vanilla_mcmc_lmls[:,0])
        # inference_fds = np.unique(vanilla_mcmc_lmls[:,1])
        #
        # # Create empty dataframe
        # vanilla_mcmc_lmls_df = pd.DataFrame(index=data_fds,columns=inference_fds)
        # # Add rows to pandas dataframe
        # for i in range(np.shape(vanilla_mcmc_lmls)[0]):
        #     vanilla_mcmc_lmls_df.loc[vanilla_mcmc_lmls[i,0], vanilla_mcmc_lmls[i,1]] = vanilla_mcmc_lmls[i,2]

        # Get list of unique data models
        data_fds = np.unique(ti_mcmc_lmls[:,0])
        inference_fds = np.unique(ti_mcmc_lmls[:,1])

        # DITTO for thermodynamic integration lmls
        # Create empty dataframe
        ti_mcmc_lmls_df = pd.DataFrame(index=data_fds,columns=inference_fds)
        # Add rows to pandas dataframe
        for i in range(np.shape(ti_mcmc_lmls)[0]):
            ti_mcmc_lmls_df.loc[ti_mcmc_lmls[i,0], ti_mcmc_lmls[i,1]] = ti_mcmc_lmls[i,2]

        # Create empty dataframe
        ti_mcmc_mean_lmls_df = pd.DataFrame(index=data_fds,columns=inference_fds)
        # Add rows to pandas dataframe
        for i in range(np.shape(ti_mcmc_mean_lmls)[0]):
            ti_mcmc_mean_lmls_df.loc[ti_mcmc_mean_lmls[i,0], ti_mcmc_mean_lmls[i,1]] = ti_mcmc_mean_lmls[i,2]

        # Compute Bayes factors
        # Copy log_marginal_likelihoods
        # vanilla_mcmc_bayes_factors_df = copy.deepcopy(vanilla_mcmc_lmls_df)
        # vanilla_mcmc_diagonal_lmls = np.diag(vanilla_mcmc_bayes_factors_df)
        # # Loop through rows
        # for i in range(vanilla_mcmc_bayes_factors_df.shape[0]):
        #     # Perform row-wise substraction of diagonal
        #     vanilla_mcmc_bayes_factors_df.iloc[i,:] = vanilla_mcmc_bayes_factors_df.iloc[i,:].apply(lambda x: subtract_lmls(x,vanilla_mcmc_diagonal_lmls[i]))

        # Compute Bayes factors
        ti_mcmc_bayes_factors_df = copy.deepcopy(ti_mcmc_lmls_df)
        ti_mcmc_diagonal_lmls = np.diag(ti_mcmc_bayes_factors_df)
        # Loop through rows
        for i in range(ti_mcmc_bayes_factors_df.shape[0]):
            # Perform row-wise substraction of diagonal
            ti_mcmc_bayes_factors_df.iloc[i,:] = ti_mcmc_bayes_factors_df.iloc[i,:].apply(lambda x: subtract_lmls(x,ti_mcmc_diagonal_lmls[i]))

        # if prints:
        #     print('Posterior Harmonic Mean marginal likelihood estimator (mu +/- var)')
        #     print(vanilla_mcmc_lmls_df)
        #     print("\n")
        #     print('Thermodynamic Integral marginal likelihood estimator (mu +/- var)')
        #     print(ti_mcmc_lmls_df)
        #     print("\n")
        #
        #     print('Posterior Harmonic Mean estimated Bayes factors (mu +/- var)')
        #     print(vanilla_mcmc_bayes_factors_df)
        #     print("\n")
        #     print('Thermodynamic Integral estimated Bayes factors (mu +/- var)')
        #     print(ti_mcmc_bayes_factors_df)
        #     print("\n")

        # Prepare export file
        filename = utils.prepare_output_experiment_summary_filename(self.experiment_metadata['id'])

        # Export to file
        if bool(export):
            # CAREFUL: experiment_id is based on the last inference id
            # vanilla_mcmc_lmls_df.to_csv(filename+f'posterior_harmonic_mean_marginal_likelihoood_estimator_{experiment_id}.csv')
            # vanilla_mcmc_bayes_factors_df.to_csv(filename+f'posterior_harmonic_mean_estimated_bayes_factors_{experiment_id}.csv')
            ti_mcmc_lmls_df.to_csv(filename+f'thermodynamic_integral_marginal_likelihoood_estimator_{experiment_id}.csv')
            ti_mcmc_mean_lmls_df.to_csv(filename+f'thermodynamic_integral_mean_marginal_likelihoood_estimator_{experiment_id}.csv')
            ti_mcmc_bayes_factors_df.to_csv(filename+f'thermodynamic_integral_estimated_bayes_factors_{experiment_id}.csv')


    def compile_sensitivity_analysis_marginal_likelihood_matrix(self,data_ids,inference_ids,experiment_id,inference_method='grwmh',prints:bool=False,export:bool=False):

        # Initialise list of rows of log marginal likelihoods
        sensitivities = ['diffuse','regular','informative']

        # Initialise result array
        ti_mcmc_lmls = []

        # Get data FDs
        data_fds = np.array([d.split('_fd')[0] for d in data_ids])
        # Get unique data FDs
        data_fds = np.unique(data_fds)
        # Get model FDs
        model_fds = [i.split('_model')[0].split('_')[1] for i in inference_ids]

        # Loop through data ids
        for data_id in np.unique(data_ids):#tqdm(data_ids):
            # Get data FD
            data_fd = data_id.split('_fd',1)[0]
            # Import data simulation parameter
            data_params = utils.import_simulation_metadata(data_id)

            # Get flag for whether data is simulation
            simulation_data = bool(strtobool(data_params['simulation_flag']))
            experiment_type = "n" + str(data_params['id'].split("_n")[1])

            # Create inference ids
            if simulation_data:
                inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_sim_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]
            else:
                inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]


            # Loop through constructed inference ids
            for i,inference_id in tqdm(enumerate(inference_ids)):

                # Table entries
                ti_mcmc_lml_entry = []

                # Loop through sensitivity levels
                for sensitivity in sensitivities:

                    # Append sensitivity level to inference id
                    inference_id = inference_ids[i] + '_' + sensitivity + '_prior'
                    # Get inference model name
                    inference_fd = model_fds[i]

                    # Define experiment metadata filename
                    metadata_filename = utils.prepare_output_experiment_inference_filename(experiment_id=self.experiment_metadata['id'],inference_id=inference_id,dataset=data_id,method=inference_method)

                    # print('metadata_filename',metadata_filename)
                    # sys.exit(1)

                    # Make sure file exists
                    if not os.path.exists((metadata_filename+'metadata.json')):
                        # if prints: print(f"Metadata file {metadata_filename}metadata.json not found")
                        # vanilla_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                        ti_mcmc_lml_entry.append('nan')
                        continue


                    #  Import metadata where acceptance is part of metadata
                    with open((metadata_filename+'metadata.json')) as json_file:
                        inference_metadata = json.load(json_file)

                    # print(inference_metadata)
                    # sys.exit(1)


                    if 'thermodynamic_integration_mcmc' in inference_metadata['results'].keys():

                        # print('data_fd',data_fd)
                        # print('inference_fd',inference_fd)
                        # print('metadata_filename',metadata_filename)

                        # Get convergence flag and check that Gelman and Rubin-inferred burnin is lower than used burnin
                        # ti_mcmc_converged = all([bool(inference_metadata['results']['thermodynamic_integration_mcmc']['converged']),
                        #                         int(inference_metadata['results']['thermodynamic_integration_mcmc']['burnin']) <= int(inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin']),
                        #                         float(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate']) >= min_acceptance,
                        #                         float(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate']) <= max_acceptance])
                        ti_mcmc_converged = all([bool(inference_metadata['results']['thermodynamic_integration_mcmc']['converged']),
                                                int(inference_metadata['results']['thermodynamic_integration_mcmc']['burnin']) <= int(inference_metadata['inference']['thermodynamic_integration_mcmc']['burnin'])])

                        if not ti_mcmc_converged:
                            print('Thermodynamic Integration mcmc inference fd:',inference_fd,'data fd:',data_fd,"sensitivity:",sensitivity)
                            print('acceptance rate:',json.dumps(inference_metadata['results']['thermodynamic_integration_mcmc']['acceptance_rate'],indent=2))
                            print('burnin:',json.dumps(inference_metadata['results']['thermodynamic_integration_mcmc']['burnin'],indent=2))
                            print('\n')
                        # Add log marginal likelihood mean and var to records only if convergence was achieved
                        if ti_mcmc_converged:
                            # Get log marginal likelihood mean variance for thermodynamic integration MCMC
                            ti_lml_mean = np.round(float(inference_metadata['results']['thermodynamic_integration_mcmc']['log_marginal_likelihoods_mean']),2)
                            ti_lml_var = np.round(float(inference_metadata['results']['thermodynamic_integration_mcmc']['log_marginal_likelihoods_var']),2)
                            # Compute them into a string
                            ti_mcmc_lml_entry.append((str(ti_lml_mean)+' +/- '+str(ti_lml_var)))
                        else:
                            ti_mcmc_lml_entry.append('tuning_problem')
                    else:
                        ti_mcmc_lml_entry.append('nan')

                # Convert list entry to str
                ti_mcmc_lml_entry_str = ','.join(ti_mcmc_lml_entry)
                # print(ti_mcmc_lml_entry_str)
                # print('\n')
                # Append entry to results
                ti_mcmc_lmls.append([data_fd.capitalize(),inference_fd.capitalize(),ti_mcmc_lml_entry_str])
                # sys.exit(1)

        # Convert to np array
        # vanilla_mcmc_lmls = np.array(vanilla_mcmc_lmls)
        ti_mcmc_lmls = np.array(ti_mcmc_lmls)

        # Get list of unique data models
        data_fds = np.unique(ti_mcmc_lmls[:,0])
        inference_fds = np.unique(ti_mcmc_lmls[:,1])

        # Create empty dataframe
        ti_mcmc_lmls_df = pd.DataFrame(index=data_fds,columns=inference_fds)
        # Add rows to pandas dataframe
        for i in range(np.shape(ti_mcmc_lmls)[0]):
            ti_mcmc_lmls_df.loc[ti_mcmc_lmls[i,0], ti_mcmc_lmls[i,1]] = ti_mcmc_lmls[i,2]

        # # Compute Bayes factors
        # ti_mcmc_bayes_factors_df = copy.deepcopy(ti_mcmc_lmls_df)
        # ti_mcmc_diagonal_lmls = np.diag(ti_mcmc_bayes_factors_df)
        # # Loop through rows
        # for i in range(ti_mcmc_bayes_factors_df.shape[0]):
        #     # Perform row-wise substraction of diagonal
        #     ti_mcmc_bayes_factors_df.iloc[i,:] = ti_mcmc_bayes_factors_df.iloc[i,:].apply(lambda x: subtract_lmls(x,ti_mcmc_diagonal_lmls[i]))

        # Prepare export file
        filename = utils.prepare_output_experiment_summary_filename(self.experiment_metadata['id'])

        # print(ti_mcmc_lmls_df)
        # Export to file
        if export:
            # CAREFUL: experiment_type is based on the last inference id
            ti_mcmc_lmls_df.to_csv(filename+f'sensitivity_analysis_thermodynamic_integral_marginal_likelihoood_estimator_{experiment_id}.csv')

            # ti_mcmc_bayes_factors_df.to_csv(filename+f'sensitivity_analysis_thermodynamic_integral_estimated_bayes_factors_{experiment_type}.csv')


    def compile_r2_matrix(self,data_ids,inference_ids,experiment_id,inference_method='grwmh',prints:bool=False,export:bool=False):

        # Initialise list of rows of log marginal likelihoods
        # vanilla_mcmc_lmls = []
        r2 = []

        # Get data FDs
        data_fds = np.array([d.split('_fd')[0] for d in data_ids])
        # Get unique data FDs
        data_fds = np.unique(data_fds)
        # Get model FDs
        model_fds = [i.split('_model')[0].split('_')[1] for i in inference_ids]

        # Loop through data ids
        for data_id in np.unique(data_ids):#tqdm(data_ids):
            # Get data FD
            data_fd = data_id.split('_fd',1)[0]
            # Import data simulation parameter
            data_params = utils.import_simulation_metadata(data_id)

            # Get flag for whether data is simulation
            simulation_data = bool(strtobool(data_params['simulation_flag']))
            experiment_type = "n" + str(data_params['id'].split("_n")[1])

            # Create inference ids
            if simulation_data:
                inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_sim_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]
            else:
                # REMOVE WHEN CLEANING CODE
                if data_fd == 'm25_data':
                    inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_learn_noise_'+experiment_type+'_regular_prior') for i,m in enumerate(model_fds)]
                else:
                    inference_ids = [(inference_method+'_'+m+'_model_'+data_fd+'_learn_noise_'+experiment_type) for i,m in enumerate(model_fds)]

            # Loop through constructed inference ids
            for i,inference_id in tqdm(enumerate(inference_ids)):

                # Get inference model name
                inference_fd = model_fds[i]
                inference_id = inference_ids[i]

                # Define experiment metadata filename
                metadata_filename = utils.prepare_output_experiment_inference_filename(experiment_id=self.experiment_metadata['id'],inference_id=inference_id,dataset=data_id,method=inference_method)

                # Make sure file exists
                if not os.path.exists((metadata_filename+'metadata.json')):
                    # if prints: print(f"Metadata file {metadata_filename}metadata.json not found")
                    r2.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])
                    continue

                #  Import metadata where acceptance is part of metadata
                with open((metadata_filename+'metadata.json')) as json_file:
                    inference_metadata = json.load(json_file)

                # print('inference_id',inference_id)
                # print(json.dumps(inference_metadata['results']['vanilla_mcmc'],indent=2))

                if 'vanilla_mcmc' in list(inference_metadata['results'].keys())\
                    and 'R2' in list(inference_metadata['results']['vanilla_mcmc'].keys())\
                    and 'converged' in list(inference_metadata['results']['vanilla_mcmc'].keys()):

                    # Get convergence flag and check that Gelman and Rubin-inferred burnin is lower than used burnin
                    # vanilla_mcmc_converged = True
                    vanilla_mcmc_converged = all([float(inference_metadata['results']['vanilla_mcmc']['acceptance_rate']) >= min_acceptance,
                                            float(inference_metadata['results']['vanilla_mcmc']['acceptance_rate']) <= max_acceptance,
                                            int(inference_metadata['results']['vanilla_mcmc']['burnin']) <= int(inference_metadata['inference']['vanilla_mcmc']['burnin']),
                                            bool(inference_metadata['results']['vanilla_mcmc']['converged'])])

                    if not vanilla_mcmc_converged:
                        print('Vanilla mcmc data fd:',data_fd,'inference fd:',inference_fd)
                        print('acceptance rate:',inference_metadata['results']['vanilla_mcmc']['acceptance_rate'])
                        print('burnin:',inference_metadata['results']['vanilla_mcmc']['burnin'])
                        print('\n')
                    # Add log marginal likelihood mean and var to records only if convergence was achieved
                    if vanilla_mcmc_converged:
                        # Get R2
                        r2_entry = str(np.round(float(inference_metadata['results']['vanilla_mcmc']['R2']),2))
                    else:
                        r2_entry = 'tuning_problem'

                    # Append entry to results
                    r2.append([data_fd.capitalize(),inference_fd.capitalize(),r2_entry])
                else:
                    # print('data_id',data_id)
                    # print('inference_id',inference_id)
                    # print('\n')
                    r2.append([data_fd.capitalize(),inference_fd.capitalize(),'nan'])


        # Convert to np array
        # vanilla_mcmc_lmls = np.array(vanilla_mcmc_lmls)
        r2 = np.array(r2)

        # Get list of unique data models
        data_fds = np.unique(r2[:,0])
        inference_fds = np.unique(r2[:,1])

        # Create empty dataframe
        r2_df = pd.DataFrame(index=data_fds,columns=inference_fds)
        # Add rows to pandas dataframe
        for i in range(np.shape(r2)[0]):
            r2_df.loc[r2[i,0], r2[i,1]] = r2[i,2]

        # Prepare export file
        filename = utils.prepare_output_experiment_summary_filename(self.experiment_metadata['id'])

        # print(r2_df)
        # sys.exit(1)
        # Export to file
        if export:
            print(filename+f'R2_{experiment_id}.csv')
            # CAREFUL: experiment_type is based on the last inference id
            r2_df.to_csv(filename+f'R2_{experiment_id}.csv')


    def tune_inference(self,inference_id):

        print(f'------------------------------------Inference id: {inference_id} ------------------------------------')

        # Ensure you provide valid inputs
        if not self.valid_input(): raise ValueError(f"Cannot proceed with experiment {self.experiment_metadata['id']}")

        # Instantiate objects
        inference_model = utils.instantiate_inference_method(inference_id)
        fd = utils.instantiate_fundamental_diagram(data_id=inference_model.inference_metadata['data_id'],model=inference_model.inference_metadata['fundamental_diagram'])

        # Populate them with data
        fd.populate(experiment_id=str(self.experiment_metadata['id']))
        inference_model.populate(fd)

        # Compute MAP
        inference_model.compute_maximum_a_posteriori_estimate(prints=strtobool(self.experiment_metadata['mle']['print']))

        vanilla_converged = True
        vanilla_acceptances = [40]
        if strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['compute']):
            vanilla_thetas,vanilla_acceptances = inference_model.run_parallel_mcmc(n=3,type='vanilla_mcmc',prints=False)
            r_stat,vanilla_converged,vanilla_burnin = inference_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,prints=False)

        if not vanilla_converged or np.mean(vanilla_acceptances) < min_acceptance or np.mean(vanilla_acceptances) > max_acceptance:
            print('Vanilla MCMC')
            print('inference_id',inference_id)
            print('vanilla_burnin',vanilla_burnin)
            print('converged',vanilla_converged)
            print('acceptances',vanilla_acceptances)

        # Compute convergence criterion for Thermodynamic Integration MCMC
        ti_converged = True
        ti_acceptances = [40]
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['compute']):
            ti_thetas,ti_acceptances = inference_model.run_parallel_mcmc(n=3,type='thermodynamic_integration_mcmc',prints=False)
            r_stat,ti_converged,ti_burnin = inference_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,prints=False)

        if not ti_converged or np.mean(ti_acceptances) < min_acceptance or np.mean(ti_acceptances) > max_acceptance:
            print('Thermodynamic Integration MCMC')
            print('inference_id',inference_id)
            print('ti_burnin',ti_burnin)
            print('converged',ti_converged)
            print('acceptances',ti_acceptances)

        print("\n")
