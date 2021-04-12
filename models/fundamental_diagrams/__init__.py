import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import toml
import utils
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.optimize as so
from scipy import stats as ss
from distutils.util import strtobool
matplotlib.rc('font', **{'size'   : 18})


class FundamentalDiagram(object):

    def __init__(self,data_id):
        self.data_id = data_id

    def simulate(self,p):
        pass

    def compute_jacobian(self):
        pass

    def compute_hessian(self):
        pass

    @property
    def num_learning_parameters(self):
        return self.__num_learning_parameters

    @num_learning_parameters.setter
    def num_learning_parameters(self,num_learning_parameters):
        self.__num_learning_parameters = num_learning_parameters

    @property
    def parameter_names(self):
        return self.__parameter_names

    @parameter_names.setter
    def parameter_names(self,param_names):
        self.__parameter_names = param_names

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self,name):
        self.__name = name

    @property
    def ols_params(self):
        return self.__ols_params

    @ols_params.setter
    def ols_params(self, ols_params):
        self.__ols_params = ols_params

    @property
    def ols_q(self):
        return self.__ols_q

    @ols_q.setter
    def ols_q(self, ols_q):
        self.__ols_q = ols_q

    @property
    def rho(self):
        return self.__rho

    @rho.setter
    def rho(self, rho):
        self.__rho = rho

    @property
    def log_q(self):
        return self.__log_q

    @log_q.setter
    def log_q(self, log_q):
        self.__log_q = log_q

    @property
    def log_q_true(self):
        return self.__log_q_true

    @log_q_true.setter
    def log_q_true(self, log_q_true):
        self.__log_q_true = log_q_true

    @property
    def true_parameters(self):
        return self.__true_parameters

    @true_parameters.setter
    def true_parameters(self, true_parameters):
        self.__true_parameters = true_parameters

    @property
    def simulation_metadata(self):
        return self.__simulation_metadata

    @simulation_metadata.setter
    def simulation_metadata(self, simulation_metadata):
        self.__simulation_metadata = simulation_metadata

    def __str__(self):
        parameter_strs = []
        for i in range(len(self.parameter_names)):
            parameter_strs.append((str(self.parameter_names[i])+' = '+str(self.true_parameters[i])))
        return ", ".join(parameter_strs)

    def populate(self,experiment_id:str=''):
        # Import q and rho
        self.import_raw_data(experiment_id=experiment_id)

        # Compute OLS estimate
        self.ols_estimation()


    def populate_rho(self):
        # Make sure you have imported metadata
        utils.validate_attribute_existence(self,['simulation_metadata'])

        # Define rho
        rho_intervals = []
        for interval in self.simulation_metadata['rho']:
            rho = np.linspace(float(self.simulation_metadata['rho'][interval]['rho_min']),
                        float(self.simulation_metadata['rho'][interval]['rho_max']),
                        int(self.simulation_metadata['rho'][interval]['rho_steps']))
            # Append to intervals
            rho_intervals.append(rho)

        # Update class attribute in object
        self.rho = np.concatenate(rho_intervals)

    def store_simulation_data(self):
        # Get true parameters if they exist (i.e. data is a simulation)
        if self.simulation_flag and bool(self.simulation_metadata['true_parameters']) and (len(list(self.simulation_metadata['true_parameters'].keys())) == self.num_learning_parameters):
            self.true_parameters = np.array([float(p) for p in self.simulation_metadata['true_parameters'].values()])

            # Print parameter values
            print(self)
        else:
            # Update simulation flag in case it was set to True by mistaked in metadata
            self.simulation_flag = False
            # Set true parameters to None
            self.true_parameters = None

    def simulate_with_noise(self,p,prints:bool=False):
        # Populate rho in metadata
        self.populate_rho()

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','simulation_metadata'])

        # Fix random seed
        if self.simulation_metadata['seed'] == '' or self.simulation_metadata['seed'] == 'None': np.random.seed(None)
        else: np.random.seed(int(self.simulation_metadata['seed']))


        # Define sigma2 to be the last parameter
        sigma2 = p[-1]
        if prints: print(f'Simulating with {sigma2} variance')

        # Get dimension of rho data
        dim = self.rho.shape[0]

        # Simulate log data without noise using log parameters
        self.log_q_true = self.log_simulate(p)

        # Generate error
        error = np.random.normal(loc=0,scale=np.sqrt(sigma2),size=len(self.log_q_true))

        # Simulate log data from multivariate normal distribution
        if strtobool(self.simulation_metadata['error']['multiplicative']):
            # Multiplicative error
            self.log_q = self.log_q_true + error
        else:
            print('simulate_with_noise Might be wrong')
            # Additive error
            self.log_q = np.log(np.exp(self.log_q_true) + error)

        # q = np.array([ss.lognorm.rvs(scale = (exp_mean[i]), s = stdev,size=100000) for i in range(len(self.q_true))])
        # # noise = np.random.multivariate_normal(np.zeros(len(self.q_true)),np.eye(len(self.q_true))*sigma2,size=100000)
        # # q = self.q_true*np.exp(noise)
        # q_mean = np.mean(q,axis=1)
        # q_std = np.std(q,axis=1)

        # Update true parameters
        self.true_parameters = p

    def squared_error(self,p):
        return np.sum((np.exp(self.log_q)-np.exp(self.log_simulate(p)))**2)

    def ols_estimation(self):

        warnings.simplefilter("ignore")

        # Make sure you have imported metadata
        utils.validate_attribute_existence(self,['simulation_metadata','simulate','hessian','squared_error','rho'])

        # Get p0 from metadata
        p0 = list(self.simulation_metadata['ols']['p0'])

        # Define constraints on positivty and concavity
        positivity_constraint = so.NonlinearConstraint(self.simulate,np.ones(len(self.rho))* np.inf,np.zeros(len(self.rho)))
        concavity_constraint = so.NonlinearConstraint(self.hessian,np.ones(len(self.rho))* -np.inf,np.zeros(len(self.rho)))

        # Optimise parameters for least squares
        if self.simulation_metadata['ols']['method'] == '':
            constrained_ls = so.minimize(self.squared_error, p0, constraints=[positivity_constraint,concavity_constraint])
        else:
            constrained_ls = so.minimize(self.squared_error, p0, constraints=[positivity_constraint,concavity_constraint],method=self.simulation_metadata['ols']['method'])
        constrained_ls_params = constrained_ls.x
        constrained_ls_q_hat = self.simulate(constrained_ls_params)

        # Update class variables
        self.ols_q = constrained_ls_q_hat
        self.ols_params = constrained_ls_params

    def import_raw_data(self,experiment_id:str=''):

        if experiment_id == '':
            # Get data filename
            filename = utils.prepare_output_simulation_filename(self.data_id)
        else:
            filename = utils.prepare_output_experiment_simulation_filename(experiment_id,dataset=self.data_id)

        # Import rho
        if os.path.exists((filename+'rho.txt')):
            rho = np.loadtxt((filename+'rho.txt'))
        else:
            raise FileNotFoundError(f"File {(filename+'rho.txt')} not found.")
        # Update attribute rho
        self.rho = rho

        # Import q
        if os.path.exists((filename+'log_q.txt')):
            log_q = np.loadtxt((filename+'log_q.txt'))
        else:
            raise FileNotFoundError(f"File {(filename+'log_q.txt')} not found.")
        # Update attribute rho
        self.log_q = log_q


    def export_data(self,experiment_id:str='',prints:bool=False):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','log_q_true'])

        if experiment_id == '':
            # Get data filename
            filename = utils.prepare_output_simulation_filename(self.data_id)
        else:
            filename = utils.prepare_output_experiment_simulation_filename(experiment_id,dataset=self.data_id)

        # Export rho

        # Ensure directory exists otherwise create it
        utils.ensure_dir(filename)
        # Save to txt file
        np.savetxt((filename+'rho.txt'),self.rho)
        if prints: print(f"File exported to {(filename+'rho.txt')}")

        # Export q_true

        # Ensure directory exists otherwise create it
        utils.ensure_dir(filename)
        # Save to txt file
        np.savetxt((filename+'log_q_true.txt'),self.log_q_true)

        if prints: print(f"File exported to {(filename+'log_q_true.txt')}")

        # Export q if it exists
        if hasattr(self,'log_q'):
            # Ensure directory exists otherwise create it
            utils.ensure_dir(filename)
            # Save to txt file
            np.savetxt((filename+'log_q.txt'),self.log_q)

            if prints: print(f"File exported to {(filename+'log_q.txt')}")


    def plot_simulation(self,plot_log:bool=True):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','log_q_true'])

        fig = plt.figure(figsize=(10,10))
        if hasattr(self,'log_q'):
            if plot_log: plt.scatter(self.rho,self.log_q,color='blue',label='Log simulated')
            else: plt.scatter(self.rho,np.exp(self.log_q),color='blue',label='Simulated')
        if plot_log:
            plt.plot(self.rho,self.log_q_true,color='black',label='Log true')
            plt.ylabel(r'$\log q$')
        else:
            plt.plot(self.rho,np.exp(self.log_q_true),color='black',label='True')
            plt.ylabel(r'$q$')
        plt.xlabel(r'$\rho$')
        plt.legend()
        plt.show()
        return fig


    def export_simulation_plot(self,experiment_id:str='',plot_log:bool=True,show_plot:bool=False,prints:bool=False):

        if experiment_id == '':
            # Get data filename
            filename = utils.prepare_output_simulation_filename(self.data_id)
        else:
            filename = utils.prepare_output_experiment_simulation_filename(experiment_id,dataset=self.data_id)

        # Generate plot
        fig = self.plot_simulation(plot_log)

        # Export plot to file
        fig.savefig((filename+'log_simulation.png'))
        # Close plot
        plt.close(fig)
        if prints: print(f"File exported to {(filename+'log_simulation.png')}")
