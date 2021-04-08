import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
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


    def run(self,data_inference_pair):

        # Flatten list to populate variables
        data_id,inference_id = data_inference_pair

        # Instantiate objects
        inference_model = utils.instantiate_inference_method(inference_id)
        fd = utils.instantiate_fundamental_diagram(inference_model.inference_metadata['data_id'])

        # Assert that data id is the same in inference and simulation metadata
        # assert inference_model.inference_metadata['data_id'] == fd.simulation_metadata['id']

        # Populate them with data
        fd.populate()
        inference_model.populate(fd)

        # Compute MLE estimate
        if strtobool(self.experiment_metadata['mle']['compute']):
            inference_model.compute_maximum_likelihood_estimate(fd,prints=strtobool(self.experiment_metadata['mle']['print']))

        # Plot univariate prior distributions
        if strtobool(self.experiment_metadata['priors']['export']):
            # print('Export priors')
            inference_model.export_univariate_prior_plots(fd,
                                                        show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))
        elif strtobool(self.experiment_metadata['priors']['show_plot']):
            inference_model.generate_univariate_prior_plots(fd,
                                                        show_plot=strtobool(self.experiment_metadata['priors']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['priors']['show_title']))

        # Import/compute log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['import']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute']):
            print('Import log unnormalised posterior')
            inference_model.import_log_unnormalised_posterior(fd.parameter_names[:-1])
        elif not strtobool(self.experiment_metadata['log_unnormalised_posterior']['import']) and strtobool(self.experiment_metadata['log_unnormalised_posterior']['compute']):
            print('Compute log unnormalised posterior')
            log_true_posterior,parameters_mesh = inference_model.evaluate_log_unnormalised_posterior(fd)

        # Export/store log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['export']):
            print('Export log unnormalised posterior')
            inference_model.export_log_unnormalised_posterior(fd,
                                                            prints=strtobool(self.experiment_metadata['compute']['log_unnormalised_posterior']['print']))
            inference_model.export_log_unnormalised_posterior_plots(fd,
                                                                show_plot=strtobool(self.experiment_metadata['log_unnormalised_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['log_unnormalised_posterior']['show_title']),
                                                                prints=strtobool(self.experiment_metadata['log_unnormalised_posterior']['print']))


        # Compute convergence criterion for Vanilla MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['compute']):
            vanilla_thetas = inference_model.run_parallel_mcmc(type='vanilla_mcmc',
                                                                fundamental_diagram=fd,
                                                                prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_vanilla_mcmc(vanilla_thetas,
                                                                            prints=strtobool(self.experiment_metadata['vanilla_mcmc']['convergence_diagnostic']['print']))
        # Compute convergence criterion for Thermodynamic Integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['compute']):
            ti_thetas = inference_model.run_parallel_mcmc(type='thermodynamic_integration_mcmc',
                                                    fundamental_diagram=fd,
                                                    prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))
            inference_model.compute_gelman_rubin_statistic_for_thermodynamic_integration_mcmc(ti_thetas,
                                                                                        prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['convergence_diagnostic']['print']))

        # Run MCMC
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            # Import Vanilla MCMC chain
            print('Import Vanilla MCMC samples')
            inference_model.import_vanilla_mcmc_samples(fd)
        elif not strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            print('Run MCMC')
            theta_accepted,theta_proposed,acceptance = inference_model.vanilla_mcmc(fd,
                                                                            seed = strtobool(inference_model.inference_metadata['vanilla_mcmc']['parameter_posterior']['seed']),
                                                                            prints = strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['print']))

        # Run thermodynamic integration MCMC
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            # Import Thermodynamic Integration MCMC chain
            print('Import Thermodynamic Integration MCMC samples')
            inference_model.import_thermodynamic_integration_mcmc_samples(fd)
        elif not strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']) and strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            print('Run thermodynamic integration MCMC')
            ti_theta_accepted,ti_acceptance = inference_model.thermodynamic_integration_mcmc(fd,
                                                                                    seed = strtobool(inference_model.inference_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['seed']),
                                                                                    prints = strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['print']))

        # Export MCMC chains
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']) or strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
            print('Export MCMC samples')
            inference_model.export_mcmc_samples()


        # Export vanilla MCMC plots
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']):
                print('Export Vanilla MCMC plots')
                inference_model.export_mcmc_parameter_posterior_plots(fd,
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_mcmc_space_exploration_plots(fd,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_mcmc_mixing_plots(fd,
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_mcmc_acf_plots(fd,
                                                        show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                        show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
            elif strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']):
                    _ = inference_model.generate_mcmc_parameter_posterior_plots(fd,
                                                                    num_stds=2,
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_mcmc_space_exploration_plots(fd,
                                                                    show_posterior=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_true_posterior']),
                                                                    show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                    show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_mcmc_mixing_plots(fd,
                                                                show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
                    _ = inference_model.generate_mcmc_acf_plots(fd,
                                                                show_plot=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_plot']),
                                                                show_title=strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show_title']))
        # Export thermodynamic integration MCMC plots
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['compute']):
            if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
                print('Export thermodynamic integration MCMC plots')
                inference_model.export_thermodynamic_integration_mcmc_mixing_plots(fd,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(fd,
                                                                                    num_stds=2,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                inference_model.export_thermodynamic_integration_mcmc_space_exploration_plots(fd,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
            elif strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']):
                _ = inference_model.generate_thermodynamic_integration_mcmc_mixing_plots(fd,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_parameter_posterior_plots(fd,
                                                                                    num_stds=2,
                                                                                    show_plot=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_plot']),
                                                                                    show_title=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_title']))
                _ = inference_model.generate_thermodynamic_integration_mcmc_space_exploration_plots(fd,
                                                                                    show_posterior=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show_true_posterior']),
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
                inference_model.export_posterior_predictive()
                inference_model.export_mcmc_posterior_predictive_plot(fd,
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
        inference_model.export_metadata()


    def run_sequentially(self):

        for data_id in list(self.experiment_metadata['data_ids']):
            for inference_id in list(self.experiment_metadata['inference_ids']):
                self.run([data_id,inference_id])
