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
        fd = utils.instantiate_fundamental_diagram(data_id)
        inference_model = utils.instantiate_inference_method(inference_id)

        # Populate them with data
        fd.populate()
        inference_model.populate(fd)

        # Compute MLE estimate
        inference_model.compute_mle_estimate(fd,prints=strtobool(self.experiment_metadata['mle']['print']))

        # Plot univariate prior distributions
        if strtobool(self.experiment_metadata['priors']['export']): inference_model.export_univariate_prior_plots(fd,strtobool(self.experiment_metadata['priors']['show']))

        # Import/compute log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['import']):
            inference_model.import_log_unnormalised_posterior(fd.parameter_names[:-1])
        else:
            log_true_posterior,parameters_mesh = inference_model.evaluate_log_unnormalised_posterior(fd)

        # Export/store log unnormalised posterior
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['export_data']):
            inference_model.export_log_unnormalised_posterior(fd,prints=strtobool(self.experiment_metadata['compute']['log_unnormalised_posterior']['print']))
        if strtobool(self.experiment_metadata['log_unnormalised_posterior']['export_plot']):
            inference_model.export_log_unnormalised_posterior_plots(fd,
                                                                strtobool(self.experiment_metadata['log_unnormalised_posterior']['show']),
                                                                prints=strtobool(self.experiment_metadata['log_unnormalised_posterior']['print']))

        # Import MCMC chains
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']) or strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']):
            inference_model.import_mcmc_samples(fd)

        # Run MCMC
        if not strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['import']):
            theta_accepted,theta_proposed,acceptance = inference_model.vanilla_mcmc(fd,
                                                                            strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['print']),
                                                                            None)
        # Run thermodynamic integration MCMC
        if not strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['import']):
            ti_theta_accepted,ti_acceptance = inference_model.thermodynamic_integration_mcmc(fd,
                                                                                    strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['print']),
                                                                                    None)
        # Export MCMC chains
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']) or strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
            inference_model.export_mcmc_samples()


        # Export vanilla MCMC plots
        if strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['export']):
            inference_model.export_mcmc_parameter_posterior_plots(fd,2,strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show']))
            inference_model.export_mcmc_space_exploration_plots(fd,strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show']))
            inference_model.export_mcmc_mixing_plots(fd,strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show']))
            inference_model.export_mcmc_acf_plots(fd,strtobool(self.experiment_metadata['vanilla_mcmc']['parameter_posterior']['show']))

        # Export thermodynamic integration MCMC plots
        if strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['export']):
            inference_model.export_thermodynamic_integration_mcmc_mixing_plots(fd,strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show']))
            inference_model.export_thermodynamic_integration_mcmc_parameter_posterior_plots(fd,2,strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show']))
            inference_model.export_thermodynamic_integration_mcmc_space_exploration_plots(fd,strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['parameter_posterior']['show']))

        # Import/Compute posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['import']):
            inference_model.import_posterior_predictive()
        else:
            inference_model.evaluate_posterior_predictive_moments()

        # Expore/Store posterior predictive
        if strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['export']):
            inference_model.export_posterior_predictive()
            inference_model.export_mcmc_posterior_predictive_plot(fd,2,strtobool(self.experiment_metadata['vanilla_mcmc']['posterior_predictive']['show']))

        # Marginal likelihood estimators

        # Compute Vanilla MCMC marginal likelihood estimator
        if not strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['import']):
            inference_model.compute_log_posterior_harmonic_mean_estimator(prints=strtobool(self.experiment_metadata['vanilla_mcmc']['marginal_likelihood']['print']))

        # Compute thermodynamic integration MCMC marginal likelihood estimator
        if not strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['import']):
            inference_model.compute_thermodynamic_integration_log_marginal_likelihood_estimator(prints=strtobool(self.experiment_metadata['thermodynamic_integration_mcmc']['marginal_likelihood']['print']))

        # Export metadata
        inference_model.export_metadata()


    def run_sequentially(self):

        for data_id in list(self.experiment_metadata['data_ids']):
            for inference_id in list(self.experiment_metadata['inference_ids']):
                self.run([data_id,inference_id])
