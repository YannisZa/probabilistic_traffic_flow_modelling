import os, sys
import math
import toml
import json
import collections.abc
import scipy.stats as ss

from fundamental_diagrams.fundamental_diagram_definitions import *
from inference.mcmc_inference_models import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]


def gaussian(p,loc,scale,transformation:str='identity'):
    if transformation == 'log': return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*((np.exp(p)-loc)/scale)**2 + p, loc, scale
    elif transformation == 'reciprocal': return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*((1./p-loc)/scale)**2 + 2*p, loc, scale
    elif transformation == 'identity': return -(1/2)*np.log(2*np.pi*(scale**2)) - (1/2)*((p-loc)/scale)**2, loc, scale
    else: raise ValueError(f'No parameter transformation found for {transformation.lower()}')

def multivariate_gaussian(p,loc,scale):
    try:
        assert p.shape[0] == loc.shape[0] and loc.shape[0] == scale.shape[0] and scale.shape[0] == scale.shape[1]
    except:
        raise ValueError(f'List lengths p {p.shape[0]}, loc {loc.shape[0]}, scale {scale.shape[0]}x{scale.shape[1]} do not match.')

    raise ValueError('Not implemented yet')
    # return -(n/2)*np.log(2*np.pi*scale**2) -(1/(2*scale**2)) * (p-loc).T @ (p-loc)


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
