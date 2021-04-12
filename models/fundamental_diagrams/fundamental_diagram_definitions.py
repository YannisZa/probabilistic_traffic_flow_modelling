import numpy as np
from fundamental_diagrams import FundamentalDiagram


class ExponentialFD(FundamentalDiagram):
    pass

    def __init__(self,data_id):
        super().__init__(data_id)
        FundamentalDiagram.name.fset(self, 'exponential')
        FundamentalDiagram.num_learning_parameters.fset(self, 3)
        FundamentalDiagram.parameter_names.fset(self, [r'$\alpha$',r'$\beta$',r'$\sigma^2$'])

    def simulate(self,p):
        return p[0]*super().rho*np.exp(-p[1]*super().rho)

    def log_simulate(self,p):
        return np.log(p[0])+np.log(super().rho)-p[1]*super().rho

    def log_simulate_with_x(self,p,rho):
        return np.log(p[0])+np.log(rho)-p[1]*rho

    def hessian(self,p):
        return -p[0]*p[1]*np.exp(-p[1]*super().rho) - p[0]*np.exp(-p[1]*super().rho) + p[0]*super().rho*np.exp(-p[1]*super().rho)


class GreenshieldsFD(FundamentalDiagram):
    pass

    def __init__(self,data_id):
        super().__init__(data_id)
        FundamentalDiagram.name.fset(self, 'greenshields')
        FundamentalDiagram.num_learning_parameters.fset(self, 3)
        FundamentalDiagram.parameter_names.fset(self, [r'$u_0$',r'$\rho_j$',r'$\sigma^2$'])

    def simulate(self,p):
        return p[0] * super().rho * (1 - super().rho/p[1])

    def log_simulate(self,p):
        return np.log(p[0])+np.log(super().rho)+np.log(1-super().rho/p[1])

    def log_simulate_with_x(self,p,rho):
        return np.log(p[0])+np.log(rho)+np.log(1-rho/p[1])

    def hessian(self,p):
        return [-2*p[0]/p[1] for i in range(len(super().rho))]
