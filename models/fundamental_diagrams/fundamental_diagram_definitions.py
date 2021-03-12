import numpy as np
from fundamental_diagrams import FundamentalDiagram


class ExponentialFD(FundamentalDiagram):
    pass

    def __init__(self):
        super().__init__('exponential')
        FundamentalDiagram.parameter_number.fset(self, 2)
        FundamentalDiagram.parameter_names.fset(self, [r'$\alpha$',r'$\beta$'])

    def simulate(self,p):
        FundamentalDiagram.q_true.fset(self, p[0]*super().rho*np.exp(-p[1]*super().rho))
        return super().q_true

    def simulate_with_x(self,p,rho):
        return p[0]*rho*np.exp(-p[1]*rho)

    def compute_jacobian(self,p):
        FundamentalDiagram.jacobian.fset(self, p[0]*np.exp(-p[1]*super().rho) - p[0]*super().rho*np.exp(-p[1]*super().rho))
        return super().jacobian

    def compute_hessian(self,p):
        FundamentalDiagram.hessian.fset(self, -p[0]*p[1]*np.exp(-p[1]*super().rho) - p[0]*np.exp(-p[1]*super().rho) + p[0]*super().rho*np.exp(-p[1]*super().rho))
        return super().hessian
