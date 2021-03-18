import numpy as np
from fundamental_diagrams import FundamentalDiagram


class ExponentialFD(FundamentalDiagram):
    pass

    def __init__(self,data_id):
        super().__init__(data_id)
        FundamentalDiagram.name.fset(self, 'exponential')
        FundamentalDiagram.num_learning_parameters.fset(self, 2)
        FundamentalDiagram.parameter_names.fset(self, [r'$\alpha$',r'$\beta$','r$\sigma$'])

    def simulate(self,p):
        return p[0]*super().rho*np.exp(-p[1]*super().rho)

    def simulate_with_x(self,p,rho):
        return p[0]*rho*np.exp(-p[1]*rho)

    def jacobian(self,p):
        return p[0]*np.exp(-p[1]*super().rho) - p[0]*super().rho*np.exp(-p[1]*super().rho)

    def hessian(self,p):
        return -p[0]*p[1]*np.exp(-p[1]*super().rho) - p[0]*np.exp(-p[1]*super().rho) + p[0]*super().rho*np.exp(-p[1]*super().rho)
