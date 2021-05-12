import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
import matplotlib.pyplot as plt
from inference.mcmc_inference_models import MetropolisHastings
from distutils.util import strtobool

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define experiment id
inference_id = str(sys.argv[1])

# Instantiate objects
inf_model = utils.instantiate_inference_method(inference_id)
fd = utils.instantiate_fundamental_diagram(data_id=inf_model.inference_metadata['data_id'],model=inf_model.inference_metadata['fundamental_diagram'])

print("Inference id:",inference_id)
print("Data id:",inf_model.inference_metadata['data_id'])

# Populate them with data
fd.populate()
inf_model.populate(fd)

# Compute MLE estimate
inf_model.compute_maximum_a_posteriori_estimate(prints=True)
