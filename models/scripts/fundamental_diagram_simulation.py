import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toml
import utils
from inference import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define path to experiment parameters
data_id = 'exponential_fd_simulation'

# Import emtadata from file
metadata = utils.import_simulation_metadata(data_id)

# Instantiate specified Fundamental Diagram
fd = utils.map_name_to_class(metadata['fundamental_diagram'])

# Setup
fd.setup(data_id)

# Compute q based on rho and specified parameters
# fd.simulate(fd.true_parameters)
fd.simulate_with_noise(fd.true_parameters)

# Export plot and data
fd.export_simulation_plot(str(data_id))
fd.export_data(str(data_id))
