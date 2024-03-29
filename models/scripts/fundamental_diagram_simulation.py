import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toml
import utils
from inference import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define path to experiment parameters
data_id = str(sys.argv[1])
        #'exponential_fd_simulation_small'
        #'exponential_fd_simulation_smaller_more_data'
        #"exponential_fd_simulation_small_medium_noise"
        #'exponential_fd_simulation_small'
        #'exponential_fd_simulation'

print('Data id:',data_id)
# Instantiate specified Fundamental Diagram
fd = utils.instantiate_fundamental_diagram(data_id)

# Compute q based on rho and specified parameters
# fd.populate()
fd.simulate_with_noise(fd.true_parameters)

# print('true_parameters',fd.true_parameters)
# print('rho',fd.rho)
# print('log_q_true',fd.log_q_true)
# print('q',fd.log_q)

# Export plot and data
fd.export_simulation_plot(plot_log=True,show_plot=True)
fd.export_data()
