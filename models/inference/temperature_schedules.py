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
    return np.array([i/n for i in range(1,n+1)]).reshape((n,1))

# Prior geometric-based temperature schedule
def prior_temperature(n,pow):
    return np.array([(i/n)**pow for i in range(1,n+1)]).reshape((n,1))

# Posterior geometric-based temperature schedule
def posterior_temperature(n,pow):
    return np.array([(1-(i/n)**pow) for i in range(1,n+1)]).reshape((n,1))
