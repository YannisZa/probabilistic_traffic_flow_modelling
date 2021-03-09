import os
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

    @jacobian.deleter
    def jacobian(self):
        del self.__jac

    @property
    def hessian(self):
        return self.__hess

    @hessian.setter
    def hessian(self,hess):
        self.__hess = hess

    @hessian.deleter
    def hessian(self):
        del self.__hess

    @property
    def rho(self):
        return self.__rho

    @rho.setter
    def rho(self, rho):
        self.__rho = rho

    @rho.deleter
    def rho(self):
        del self.__rho

    @property
    def q(self):
        return self.__q

    @q.setter
    def q(self, q):
        self.__q = q

    @q.deleter
    def q(self):
        del self.__q

    @property
    def q_true(self):
        return self.__q_true

    @q_true.setter
    def q_true(self, q_true):
        self.__q_true = q_true

    @q_true.deleter
    def q_true(self):
        del self.__q_true

    def simulate_with_noise(self,p,sigma2,seed:float=None):
        # Make sure you have stored the necessary attributes
        if not hasattr(self,'rho'):
            raise AttributeError('Rho attribute does not exist.')

        # Fix random seed
        np.random.seed(seed)
        # Get dimension of rho data
        dim = self.rho.shape[0]
        # Simulate without noise
        self.simulate(p)
        # Generate white noise
        noise = np.random.multivariate_normal(np.zeros(dim),np.eye(dim)*sigma2)
        # Update q (noisy) and q_true (noise-free)
        self.q = self.q_true*np.exp(noise)

    def import_data(self,filename):

        # Import rho
        if os.path.exists((filename+'_rho.txt')):
            rho = np.loadtxt((filename+'_rho.txt'))
        else:
            raise FileNotFoundError(f"File {(filename+'_rho.txt')} not found.")
        # Update attribute rho
        self.rho = rho

        # Import q
        if os.path.exists((filename+'_q.txt')):
            q = np.loadtxt((filename+'_q.txt'))
        else:
            raise FileNotFoundError(f"File {(filename+'_rho.txt')} not found.")
        # Update attribute rho
        self.q = q

    def export_data(self,filename):

        # Export rho
        if hasattr(self,'rho'):
            np.savetxt((filename+'_rho.txt'),self.rho)
            print(f"File exported to {(filename+'_rho.txt')}")
        else:
            raise AttributeError('Rho attribute does not exist.')
        # Export q
        if hasattr(self,'q'):
            np.savetxt((filename+'_q.txt'),self.q)
            print(f"File exported to {(filename+'_q.txt')}")
        else:
            raise AttributeError('q attribute does not exist.')


    def plot_simulation(self):

        # Make sure you have stored the necessary attributes
        if not hasattr(self,'rho'):
            raise AttributeError('Rho attribute does not exist.')
        if not hasattr(self,'q_true'):
            raise AttributeError('q_true attribute does not exist.')
        if not hasattr(self,'q'):
            raise AttributeError('q attribute does not exist.')

        fig = plt.figure(figsize=(10,10))
        plt.scatter(self.rho,self.q,color='blue',label='Simulated')
        plt.plot(self.rho,self.q_true,color='red',label='True')
        plt.xlabel(r'$\rho$')
        plt.ylabel(r'$q$')
        plt.legend()
        plt.show()
        return fig


    def export_simulation_plot(self,filename):

        # Make sure you have stored the necessary attributes
        if not hasattr(self,'rho'):
            raise AttributeError('Rho attribute does not exist.')
        if not hasattr(self,'q'):
            raise AttributeError('q attribute does not exist.')

        # Generate plot
        fig = self.plot_simulation()

        # Export plot to file
        fig.savefig((filename+'.png'))
        print(f"File exported to {(filename+'.png')}")
