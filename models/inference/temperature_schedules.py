import numpy as np

def map_name_to_temperature_schedule(name):

    if 'uniform' in name:
        return uniform_temperature
    elif 'prior' in name:
        return prior_temperature
    elif 'posterior' in name:
        return posterior_temperature
    else:
        raise Exception(f'No fundamental diagram model found for {name.lower()}')


# Uniform geometric-based temperature schedule
def uniform_temperature(n,pow:int=1):
    return np.array([i/(n-1) for i in range(0,n)]).reshape((n))

# Prior geometric-based temperature schedule
def prior_temperature(n,pow):
    return np.array([(i/(n-1))**pow for i in range(0,n)]).reshape((n))

# Posterior geometric-based temperature schedule
def posterior_temperature(n,pow):
    return np.array([(1-(i/(n-1))**pow) for i in range(0,n)]).reshape((n))
