import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import os
import toml
import utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rc('font', **{'size'   : 18})


class FundamentalDiagram(object):

    def __init__(self,name):
        self.name = name

    def simulate(self,p):
        pass

    def compute_jacobian(self):
        pass

    def compute_hessian(self):
        pass

    @property
    def parameter_number(self):
        return self.__parameter_number

    @parameter_number.setter
    def parameter_number(self,param_num):
        self.__parameter_number = param_num

    @property
    def parameter_names(self):
        return self.__parameter_names

    @parameter_names.setter
    def parameter_names(self,param_names):
        self.__parameter_names = param_names

    @property
    def jacobian(self):
        return self.__jac

    @jacobian.setter
    def jacobian(self,jac):
        self.__jac = jac

    @property
    def hessian(self):
        return self.__hess

    @hessian.setter
    def hessian(self,hess):
        self.__hess = hess

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
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, seed):
        self.__seed = seed

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


    def setup(self,data_id):
        # Import metadata to FD object
        self.import_simulation_metadata(data_id)

        # Define rho
        rho = np.linspace(float(self.simulation_metadata['rho']['rho_min']),
                        float(self.simulation_metadata['rho']['rho_max']),
                        int(self.simulation_metadata['rho']['rho_steps']))

        # Update rho attribute in object
        self.rho = rho


    def simulate_with_noise(self,p):
        # Make sure you have stored the necessary attributes
        utils.has_attributes(self,['rho','simulation_metadata'])

        # Fix random seed
        if self.simulation_metadata['seed'] == '':np.random.seed(None)
        else: np.random.seed(int(self.simulation_metadata['seed']))

        # Define sigma2 to be the last parameter
        sigma2 = p[-1]
        print(f'Simulating with {sigma2} variance')
        # Get dimension of rho data
        dim = self.rho.shape[0]
        # Simulate without noise
        self.simulate(p)
        # Generate white noise
        noise = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*sigma2)
        # Update q (noisy) and q_true (noise-free)
        self.q = self.q_true*np.exp(noise)
        # Update true parameters
        self.true_parameters = p

    def import_data(self,data_id):

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(data_id)

        # Import rho
        if os.path.exists((data_filename+'_rho.txt')):
            rho = np.loadtxt((data_filename+'_rho.txt'))
        else:
            raise FileNotFoundError(f"File {(data_filename+'_rho.txt')} not found.")
        # Update attribute rho
        self.rho = rho

        # Import q
        if os.path.exists((data_filename+'_q.txt')):
            q = np.loadtxt((data_filename+'_q.txt'))
        else:
            raise FileNotFoundError(f"File {(data_filename+'_rho.txt')} not found.")
        # Update attribute rho
        self.q = q

    def import_simulation_metadata(self,data_id):

        # Get data filename
        metadata_filename = utils.prepare_input_simulation_filename(data_id)

        # Import simulation metadata
        if os.path.exists(metadata_filename):
            _simulation_metadata  = toml.load(metadata_filename)
        else:
            raise FileNotFoundError(f'File {metadata_filename} not found.')

        # Add to class attribute
        self.simulation_metadata = _simulation_metadata
        self.true_parameters = np.array([float(p) for p in _simulation_metadata['true_parameters'].values()])



    def export_data(self,data_id):

        # Make sure you have stored the necessary attributes
        utils.has_attributes(self,['rho','q_true'])

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(data_id)

        # Export rho

        # Ensure directory exists otherwise create it
        utils.ensure_dir(data_filename)
        # Save to txt file
        np.savetxt((data_filename+'_rho.txt'),self.rho)
        print(f"File exported to {(data_filename+'_rho.txt')}")

        # Export q_true

        # Ensure directory exists otherwise create it
        utils.ensure_dir(data_filename)
        # Save to txt file
        np.savetxt((data_filename+'_q_true.txt'),self.q_true)
        print(f"File exported to {(data_filename+'_q_true.txt')}")

        # Export q if it exists
        if hasattr(self,'q'):
            # Ensure directory exists otherwise create it
            utils.ensure_dir(data_filename)
            # Save to txt file
            np.savetxt((data_filename+'_q.txt'),self.q)
            print(f"File exported to {(data_filename+'_q.txt')}")


    def plot_simulation(self):

        # Make sure you have stored the necessary attributes
        utils.has_attributes(self,['rho','q_true'])

        fig = plt.figure(figsize=(10,10))
        if hasattr(self,'q'):
            plt.scatter(self.rho,self.q,color='blue',label='Simulated')
        plt.plot(self.rho,self.q_true,color='red',label='True')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$q$')
        plt.legend()
        plt.show()
        return fig


    def export_simulation_plot(self,data_id):

        # Get data filename
        data_filename = utils.prepare_output_simulation_filename(data_id)

        # Generate plot
        fig = self.plot_simulation()

        # Export plot to file
        fig.savefig((data_filename+'.png'))
        # Close plot
        plt.close(fig)
        print(f"File exported to {(data_filename+'.png')}")
