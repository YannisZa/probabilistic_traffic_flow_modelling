import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import toml
import utils
import numpy as np
from experiments import Experiment
from distutils.util import strtobool

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define simulation id
experiment_id = "generate_data"

# Instantiate experiment
experiment = Experiment(experiment_id)

# Run experiment
experiment.run_sequentially()
