import numpy as np
from fundamental_diagrams import FundamentalDiagram


class ExponentialFD(FundamentalDiagram):
    pass

    def __init__(self):
        super().__init__('exponential')

    def simulate(self,p):
        FundamentalDiagram.q.fset(self, p[0]*super().rho*np.exp(-p[1]*super().rho))

    def compute_jacobian(self,p):
        FundamentalDiagram.jacobian.fset(self, p[0]*np.exp(-p[1]*super().rho) - p[0]*super().rho*np.exp(-p[1]*super().rho))

    def compute_hessian(self,p):
        FundamentalDiagram.hessian.fset(self, -p[0]*p[1]*np.exp(-p[1]*super().rho) - p[0]*np.exp(-p[1]*super().rho) + p[0]*super().rho*np.exp(-p[1]*super().rho))
