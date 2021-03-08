import numpy as np
from inference import MarkovChainMonteCarlo


class GaussianRandomWalkMetropolisHastings(MarkovChainMonteCarlo):
    pass

    def __init__(self):
        super().__init__('grwmh_mcmc')

    def simulate(self,p):
        MarkovChainMonteCarlo.q.fset(self, p[0]*super().rho*np.exp(-p[1]*super().rho))

    def compute_jacobian(self,p):
        MarkovChainMonteCarlo.jacobian.fset(self, p[0]*np.exp(-p[1]*super().rho) - p[0]*super().rho*np.exp(-p[1]*super().rho))

    def compute_hessian(self,p):
        MarkovChainMonteCarlo.hessian.fset(self, -p[0]*p[1]*np.exp(-p[1]*super().rho) - p[0]*np.exp(-p[1]*super().rho) + p[0]*super().rho*np.exp(-p[1]*super().rho))
