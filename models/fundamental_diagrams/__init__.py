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
    def simulation_flag(self):
        return self.__simulation_flag

    @simulation_flag.setter
    def simulation_flag(self,simulation_flag):
        self.__simulation_flag = simulation_flag

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
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        self.__q = q

    @property
    def q_true(self):
        return self.__q_true

    @q_true.setter
    def q_true(self, q_true):
        self.__q_true = q_true

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


    def populate(self):
        # Import metadat
        self.import_metadata()

        # Import q and rho
        self.import_raw_data()

        # Compute OLS estimate
        self.ols_estimation()


    def populate_rho(self):

        # Import metadat
        self.import_metadata()

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

    def import_metadata(self):
        # Get flag of whether data are a simulation or not
        self.simulation_flag = strtobool(self.simulation_metadata['simulation_flag'])

        # Get true parameters if they exist (i.e. data is a simulation)
        if self.simulation_flag and bool(self.simulation_metadata['true_parameters']) and (len(list(self.simulation_metadata['true_parameters'].keys())) == self.num_learning_parameters + 1):
            self.true_parameters = np.array([float(p) for p in self.simulation_metadata['true_parameters'].values()])

            print('True parameters')
            for i,pname in enumerate(self.parameter_names):
                print(f'{pname} = {self.true_parameters[i]}')
        else:
            # Update simulation flag in case it was set to True by mistaked in metadata
            self.simulation_flag = False
            # Set true parameters to None
            self.true_parameters = None

    def simulate_with_noise(self,p,**kwargs):

        # Populate rho in metadata
        self.populate_rho()

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','simulation_metadata'])

        # Fix random seed
        if self.simulation_metadata['seed'] == '':np.random.seed(None)
        else: np.random.seed(int(self.simulation_metadata['seed']))

        # Define sigma2 to be the last parameter
        sigma2 = p[-1]
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f'Simulating with {sigma2} variance')
        # Get dimension of rho data
        dim = self.rho.shape[0]
        # Simulate without noise
        self.q_true = self.simulate(p)
        # Generate white noise
        # noise = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*sigma2)
        # Update q (noisy) and q_true (noise-free)
        # self.q = self.q_true*np.exp(noise)
        self.q = np.array([np.random.lognormal(mean = np.log(self.q_true[i]), sigma = np.sqrt(sigma2)) for i in range(len(self.q_true))])
        # Update true parameters
        self.true_parameters = p

    def squared_error(self,p):
        return np.sum((self.q-self.simulate(p))**2)

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

    def import_raw_data(self):

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(self.data_id)

        # Import rho
        if os.path.exists((data_filename+'rho.txt')):
            rho = np.loadtxt((data_filename+'rho.txt'))
        else:
            raise FileNotFoundError(f"File {(data_filename+'rho.txt')} not found.")
        # Update attribute rho
        self.rho = rho

        # Import q
        if os.path.exists((data_filename+'q.txt')):
            q = np.loadtxt((data_filename+'q.txt'))
        else:
            raise FileNotFoundError(f"File {(data_filename+'rho.txt')} not found.")
        # Update attribute rho
        self.q = q


    def export_data(self,**kwargs):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','q_true'])

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(self.data_id)

        # Export rho

        # Ensure directory exists otherwise create it
        utils.ensure_dir(data_filename)
        # Save to txt file
        np.savetxt((data_filename+'rho.txt'),self.rho)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(data_filename+'rho.txt')}")

        # Export q_true

        # Ensure directory exists otherwise create it
        utils.ensure_dir(data_filename)
        # Save to txt file
        np.savetxt((data_filename+'q_true.txt'),self.q_true)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(data_filename+'q_true.txt')}")

        # Export q if it exists
        if hasattr(self,'q'):
            # Ensure directory exists otherwise create it
            utils.ensure_dir(data_filename)
            # Save to txt file
            np.savetxt((data_filename+'q.txt'),self.q)
            if 'prints' in kwargs:
                if kwargs.get('prints'): print(f"File exported to {(data_filename+'q.txt')}")


    def plot_simulation(self):

        # Make sure you have stored the necessary attributes
        utils.validate_attribute_existence(self,['rho','q_true'])

        fig = plt.figure(figsize=(10,10))
        if hasattr(self,'q'):
            plt.scatter(self.rho,self.q,color='blue',label='Simulated')
        plt.plot(self.rho,self.q_true,color='red',label='True')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$q$')
        plt.legend()
        plt.show()
        return fig


    def export_simulation_plot(self,show_plot:bool=False,**kwargs):

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(self.data_id)

        # Generate plot
        fig = self.plot_simulation()

        # Export plot to file
        fig.savefig((data_filename+'simulation.png'))
        # Close plot
        plt.close(fig)
        if 'prints' in kwargs:
            if kwargs.get('prints'): print(f"File exported to {(data_filename+'simulation.png')}")
