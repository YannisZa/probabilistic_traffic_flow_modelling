class FundamentalDiagram(object):

    def __init__(self,name):
        self.name = name

    def simulate(self):
        pass

    def simulate(self):
        pass

    def compute_jacobian(self):
        pass

    def compute_hessian(self):
        pass

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
