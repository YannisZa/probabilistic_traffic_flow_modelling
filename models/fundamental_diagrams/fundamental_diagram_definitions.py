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
        FundamentalDiagram.parameter_names.fset(self, [r'$v_f$',r'$\rho_j$',r'$\sigma^2$'])

    def simulate(self,p):
        return p[0] * super().rho * (1 - super().rho/p[1])

    def log_simulate(self,p):
        return np.log(p[0])+np.log(super().rho)+np.log(1-super().rho/p[1])

    def log_simulate_with_x(self,p,rho):
        return np.log(p[0])+np.log(rho)+np.log(1-rho/p[1])

    def hessian(self,p):
        return [-2*p[0]/p[1] for i in range(len(super().rho))]

class DaganzosFD(FundamentalDiagram):
    pass

    def __init__(self,data_id):
        super().__init__(data_id)
        FundamentalDiagram.name.fset(self, 'daganzos')
        FundamentalDiagram.num_learning_parameters.fset(self, 4)
        FundamentalDiagram.parameter_names.fset(self, [r'$q_c$',r'$\rho_c$',r'$\rho_j$',r'$\sigma^2$'])

    def simulate(self,p):
        return np.array([(p[0]/p[1])*r if r < p[1] else p[0]*(p[2]-r)/(p[2]-p[1]) for r in super().rho])

    def log_simulate(self,p):
        return np.array([np.log(p[0])-np.log(p[1])+np.log(r) if np.log(r) < np.log(p[1]) else np.log(p[0])+np.log(p[2]-r)-np.log(p[2]-p[1]) for r in super().rho])

    def log_simulate_with_x(self,p,rho):
        return np.array([np.log(p[0])-np.log(p[1])+np.log(r) if np.log(r) < np.log(p[1]) else np.log(p[0])+np.log(p[2]-r)-np.log(p[2]-p[1]) for r in rho])

    def hessian(self,p):
        return [0 for i in range(len(super().rho))]

class DelCastillos(FundamentalDiagram):
    pass

    def __init__(self,data_id):
        super().__init__(data_id)
        FundamentalDiagram.name.fset(self, 'delcastillos')
        FundamentalDiagram.num_learning_parameters.fset(self, 5)
        FundamentalDiagram.parameter_names.fset(self, [r'$Z$',r'$u$',r'$\rho_j$',r'$\omega$',r'$\sigma^2$'])

    def simulate(self,p):
        return p[0]*( ((p[1]/p[2])*super().rho)**(-p[3]) + (1-(super().rho/p[2]))**(-p[3]) )**(-1/p[3])

    def log_simulate(self,p):
        return np.log(p[0]) - (1/p[3]) * np.log( ((p[1]/p[2])*super().rho)**(-p[3]) + (1-(super().rho/p[2]))**(-p[3]) )

    def log_simulate_with_x(self,p,rho):
        return np.log(p[0]) - (1/p[3]) * np.log( ((p[1]/p[2])*rho)**(-p[3]) + (1-(rho/p[2]))**(-p[3]) )

    def hessian(self,p):
        num = - p[0]*(p[2]**2)*(p[3]+1)*((p[1]/p[2])*super().rho)**p[3]*(1-(super().rho/p[2]))**p[3]
        denum = super().rho**2*(super().rho-p[2])**2 * ( ((p[1]/p[2])*super().rho)**(-p[3]) + (1-(super().rho/p[2]))**(-p[3]) )**(1/p[3]) * ( ((p[1]/p[2])*super().rho)**(p[3]) + (1-(super().rho/p[2]))**(p[3]) )**2
        return num/denum
