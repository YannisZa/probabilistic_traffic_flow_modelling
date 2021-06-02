import os, sys
import math
import toml
import json
import numpy as np
import collections.abc
import scipy.stats as ss
import scipy.optimize as so
from scipy.special import betaln

# from fundamental_diagrams.fundamental_diagram_definitions import *
# from inference.mcmc_inference_models import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]

def taylor_expansion_of_moments(loc,scale,N,transformation,jacobian,hessian,third_derivative,transformation_name):
    if 'log' in transformation_name.lower():
        return transformation(loc) + hessian(loc)*scale**2/2, ((jacobian(loc))**2*scale**2 + 0.5*(hessian(loc))**2*scale**4 + (jacobian(loc))*(third_derivative(loc))*scale**4)/(N**0.5)
    elif '1/' in transformation_name.lower():
        print('Taylor expansion or reciprocal variable needs checking')
        return transformation(loc) + scale**2/loc**3, scale**2 / loc**4 * N**(-0.5)
    else:
        return loc, scale * N**(-0.5)

def univariate_gaussian(p,loc,scale,transformation,plower:float=-np.inf,pupper:float=np.inf):
    if 'log' in transformation.lower():
        if np.isinf(plower) and np.isinf(pupper):
            raise ValueError('Cannot apply log transformation to unconstrained variables in gaussian.')
        elif np.isinf(pupper):
            # print('Lower bound')
            # Compute mode
            # mode = np.log(- (plower - loc) + np.sqrt((plower-loc)**2 + 4*scale**2)) - np.log(2)
            return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*( ( (np.exp(p)+plower) - loc ) / scale )**2 + p, loc, scale#, mode
        elif np.isinf(plower):
            # print('Upper bound')
            # Compute mode
            # mode = so.minimize(lambda x: (1/2)*( ( (pupper*np.exp(x))/(1+np.exp(x)) - loc ) / scale )**2 - x**pupper + 2*np.log(1+np.exp(x)), (pupper*np.exp(loc))/(1+np.exp(loc)))
            # mode = mode['x'][0]
            return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*( ( (pupper*np.exp(p))/(1+np.exp(p)) - loc ) / scale )**2 + p**pupper - 2*np.log(1+np.exp(p)) , loc, scale#, mode
        else:
            # print('Upper and lower bounds')
            # Compute mode
            # mode = so.minimize(lambda x: (1/2)*( ( (pupper*np.exp(x)+plower)/(1+np.exp(x)) - loc ) / scale )**2 - np.log(pupper-plower) - x + 2*np.log(1+np.exp(x)), (pupper*np.exp(loc)+plower)/(1+np.exp(loc)))
            # mode = mode['x'][0]
            return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*( ( (pupper*np.exp(p)+plower)/(1+np.exp(p)) - loc ) / scale )**2 + np.log(pupper-plower) + p - 2*np.log(1+np.exp(p)), loc, scale#, mode
    elif '1/' in transformation.lower():
        raise ValueError('univariate_gaussian for 1/ transformation need cheking')
        return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*((1./p-loc)/scale)**2 + 2*np.log(p), loc, scale#, loc
    else:
        return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*((p-loc)/scale)**2, loc, scale#, loc


def multivariate_gaussian(p,loc,scale):
    # try:
    #     assert p.shape[0] == loc.shape[0] and loc.shape[0] == scale.shape[0] and scale.shape[0] == scale.shape[1]
    # except:
    #     raise ValueError(f'List lengths p {p.shape[0]}, loc {loc.shape[0]}, scale {scale.shape[0]}x{scale.shape[1]} do not match.')
    # raise ValueError('multivariate_gaussian Not implemented yet')
    diagonal = np.diag(scale)
    return -(len(p)/2)*np.log(2*np.pi) -0.5*np.sum(diagonal) - 0.5 * (p-loc).T @ np.diag(1/diagonal) @ (p-loc)


def multivariate_gaussian_iid(p,loc,scale,log_data:bool=True):
    try:
        assert p.shape[0] == loc.shape[0] and loc.shape[0] == scale.shape[0] and scale.shape[0] == scale.shape[1]
    except:
        raise ValueError(f'List lengths p {p.shape[0]}, loc {loc.shape[0]}, scale {scale.shape[0]}x{scale.shape[1]} do not match.')

    # Get sigma2 from covariance matrix
    sigma2 = scale[0,0]
    # Get number of data points
    n = len(p)

    if log_data: return -(n/2)*np.log(2*np.pi*sigma2) -(1/(2*sigma2)) * (p-loc).T @ (p-loc)
    # Apply Jacobian transformation to turn log data to simple data
    else: return -(n/2)*np.log(2*np.pi*sigma2) -(1/(2*sigma2)) * (p-loc).T @ (p-loc) - np.sum(p)


def beta(p,alpha,beta):
    return (alpha-1)*np.log(p) + (beta-1)*np.log(1-p) - betaln(alpha,beta)
