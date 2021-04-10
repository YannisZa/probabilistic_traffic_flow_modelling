import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import time
import toml
import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as so

from distutils.util import strtobool
from fundamental_diagrams import FundamentalDiagram
from inference import MarkovChainMonteCarlo

matplotlib.rc('font', **{'size'   : 18})


class Experiment(object):

    def __init__(self,experiment_id):
        # Import metadata / experiment parameters
        self.experiment_metadata = utils.import_experiment_metadata(experiment_id)

    def valid_input(self):

        # Flag for proceeding with experiments
        proceed = True

        # Inference ids and simulation ids must be equal in number
        if len(list(self.experiment_metadata['inference_ids'])) != len(list(self.experiment_metadata['data_ids'])):
            proceed = False
            print(f"Inference and data ids are not equal in number {len(list(self.experiment_metadata['inference_ids']))} != {len(list(self.experiment_metadata['data_ids']))}")

        return proceed


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


    def run_inference(self,data_inference_pair):

        # Get starting time
        start = time.time()

        # Ensure you provide valid inputs
        if not self.valid_input(): raise ValueError(f"Cannot proceed with experiment {self.experiment_metadata['id']}")

        # Flatten list to populate variables
        data_id,inference_id = data_inference_pair

        # Instantiate objects
        inference_model = utils.instantiate_inference_method(inference_id)
        fd = utils.instantiate_fundamental_diagram(inference_model.inference_metadata['data_id'])

        # Populate them with data
        fd.populate(experiment_id=str(self.experiment_metadata['id']))
        inference_model.populate(fd)

        # Compute MLE estimate
        if strtobool(self.experiment_metadata['mle']['compute']):
            inference_model.compute_maximum_likelihood_estimate(prints=strtobool(self.experiment_metadata['mle']['print']))

        # Plot univariate prior distributions
        if strtobool(self.experiment_metadata['priors']['export']):
            # print('Export priors')
            inference_model.export_univariate_prior_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))
        elif strtobool(self.experiment_metadata['priors']['show_plot']):
            inference_model.generate_univariate_prior_plots(show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))

        # Import/compute log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['import']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute']):
            print('Import log unnormalised posterior')
            inference_model.import_log_unnormalised_posterior(inference_model.parameter_names)
        elif not strtobool(self.experiment_metadata['log_unnormalised_posterior']['import']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute']):
            print('Compute log unnormalised posterior')
            log_true_posterior,parameters_mesh = inference_model.evaluate_log_unnormalised_posterior()

        # Export/store log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['export']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute']):
            print('Export log unnormalised posterior')
            inference_model.export_log_unnormalised_posterior(experiment=str(self.experiment_metadata['id']),
                                                            prints=strtobool(self.experiment_metadata['log_unnormalised_posterior']['print']))
            inference_model.export_log_unnormalised_posterior_plots(experiment=str(self.experiment_metadata['id']),
                                                                show_plot=strtobool(self.experiment_metadata['log_unnormalised_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['log_unnormalised_posterior']['show_title']))


        # Compute convergence criterion for Vanilla MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['compute']):
            vanilla_thetas = inference_model.run_parallel_mcmc(type='vanilla_mcmc',
                                                                prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,
                                                                            prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
        # Compute convergence criterion for Thermodynamic Integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['compute']):
            ti_thetas = inference_model.run_parallel_mcmc(type='thermodynamic_integration_mcmc',
                                                    prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,
                                                                                        prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))

        # Run MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            # Import Vanilla MCMC chain
            print('Import Vanilla MCMC samples')
            inference_model.import_vanilla_mcmc_samples()
        elif not strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            print('Run MCMC')
            theta_accepted,acceptance = inference_model.vanilla_mcmc(i = 0,
                                                                    seed = int(inference_model.inference_metadata['inference']['vanilla_mcmc']['seed']),
                                                                    prints = strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['print']))

        # Run thermodynamic integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            # Import Thermodynamic Integration MCMC chain
            print('Import Thermodynamic Integration MCMC samples')
            inference_model.import_thermodynamic_integration_mcmc_samples()
        elif not strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            print('Run thermodynamic integration MCMC')
            ti_theta_accepted,ti_acceptance = inference_model.thermodynamic_integration_mcmc(i=0,
                                                                                    seed = int(inference_model.inference_metadata['inference']['thermodynamic_integration_mcmc']['seed']),
                                                                                    prints = strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['print']))

        # Export MCMC chains
        if (strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export'])
                and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']))\
            or (strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export'])
                and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute'])):
            print('Export MCMC samples')
            inference_model.export_mcmc_samples(experiment=str(self.experiment_metadata['id']))


        # Export vanilla MCMC plots
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            show_true_posterior = strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_true_posterior']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute'])
            if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']):
                print('Export Vanilla MCMC plots')
                inference_model.export_mcmc_parameter_posterior_plots(experiment=str(self.experiment_metadata['id']),
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_vanilla_mcmc_space_exploration_plots(experiment=str(self.experiment_metadata['id']),
                                                                    show_posterior=show_true_posterior,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_mcmc_mixing_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_mcmc_acf_plots(experiment=str(self.experiment_metadata['id']),
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
            elif strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']):
                    _ = inference_model.generate_mcmc_parameter_posterior_plots(num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_vanilla_mcmc_space_exploration_plots(show_posterior=show_true_posterior,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_mcmc_mixing_plots(show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_mcmc_acf_plots(show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
        # Export thermodynamic integration MCMC plots
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            show_true_posterior = strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_true_posterior']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute'])
            if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
                print('Export thermodynamic integration MCMC plots')
                inference_model.export_thermodynamic_integration_mcmc_mixing_plots(experiment=str(self.experiment_metadata['id']),
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(experiment=str(self.experiment_metadata['id']),
                                                                                    num_stds=2,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_thermodynamic_integration_mcmc_space_exploration_plots(experiment=str(self.experiment_metadata['id']),
                                                                                    show_posterior=show_true_posterior,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
            elif strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']):
                _ = inference_model.generate_thermodynamic_integration_mcmc_mixing_plots(show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_parameter_posterior_plots(num_stds=2,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_space_exploration_plots(show_posterior=show_true_posterior,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))

        # Import/Compute posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']):
            print('Import posterior predictive')
            inference_model.import_posterior_predictive()
        elif not strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['compute']):
            print('Compute posterior predictive')
            inference_model.evaluate_posterior_predictive_moments(prints=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['print']))

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
            elif strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_plot']):
                _ = inference_model.generate_posterior_predictive_plot(fd,
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show_title']))

        # Marginal likelihood estimators

        # Compute Vanilla MCMC marginal likelihood estimator
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['compute']):
                print('Compute posterior harmonic mean marginal likelihood estimator')
                inference_model.compute_log_posterior_harmonic_mean_estimator(prints=strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['print']))

        # Compute thermodynamic integration MCMC marginal likelihood estimator
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['compute']):
                print('Compute thermodynamic integration marginal likelihood estimator')
                inference_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['print']))

        # Export metadata
        inference_model.export_metadata(experiment=str(self.experiment_metadata['id']))

        end = time.time()
        hours, rem = divmod(end-start, 3600)
        minutes, seconds = divmod(rem, 60)
        experiment_id = str(self.experiment_metadata['id'])
        print(f"Experiment "+experiment_id+" finished in {:0>2}:{:0>2}:{:05.2f} hours...".format(int(hours),int(minutes),seconds))

    def run_sequentially(self):

        if strtobool(self.experiment_metadata['routines']['generate_data']):
            for data_id in set(list(self.experiment_metadata['data_ids'])):
                self.generate_data(data_id)

        if strtobool(self.experiment_metadata['routines']['run_inference']):
            for data_id in list(self.experiment_metadata['data_ids']):
                for inference_id in list(self.experiment_metadata['inference_ids']):
                    self.run_inference([data_id,inference_id])
