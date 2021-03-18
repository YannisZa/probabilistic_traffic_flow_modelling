import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toml
import utils
from inference import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define path to experiment parameters
data_id = "exponential_fd_simulation_small_medium_noise"
        #'exponential_fd_simulation_small'
        #'exponential_fd_simulation'

# Instantiate specified Fundamental Diagram
fd = utils.instantiate_fundamental_diagram(data_id)

# Load simulation metadata
fd.import_simulation_metadata(data_id)

# Setup
fd.setup(data_id)

# Compute q based on rho and specified parameters
# fd.simulate(fd.true_parameters)
fd.simulate_with_noise(fd.true_parameters)

# Export plot and data
fd.export_simulation_plot(str(data_id),True)
fd.export_data(str(data_id))
