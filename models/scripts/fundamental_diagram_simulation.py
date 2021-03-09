import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import toml
from utils import map_name_to_class,prepare_simulation_filename
from inference import *

# Root directory
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))).split('probabilistic_traffic_flow_modelling/')[0]+"probabilistic_traffic_flow_modelling"

# Define path to experiment parameters
simulation_parameters_filepath = os.path.join(root,'data/input/simulation_parameters/sample_simulation.toml')

# Load experiment parameters
if os.path.exists(simulation_parameters_filepath):
    simulation_parameters  = toml.load(simulation_parameters_filepath)
else:
    raise FileNotFoundError(f'File {simulation_parameters_filepath} not found.')

# Read and convert some simulation parameters
seed = simulation_parameters['simulation']['seed']
if seed == '': seed = None
else: seed = int(seed)

# Instantiate specified Fundamental Diagram
fd = map_name_to_class(simulation_parameters['fundamental_diagram'])

# Set rho
rho = np.linspace(float(simulation_parameters['simulation']['rho_min']),
                float(simulation_parameters['simulation']['rho_max']),
                int(simulation_parameters['simulation']['rho_steps']))

# Update rho attribute in object
fd.rho = rho

# Compute q based on rho and specified parameters
fd.simulate_with_noise([float(simulation_parameters['simulation']['alpha']),
                            float(simulation_parameters['simulation']['beta'])],
                            float(simulation_parameters['simulation']['sigma2']),
                            seed)

# Simulation output filename
filename = prepare_simulation_filename(simulation_parameters['id'])

# Plot simulation
# fd.plot_simulation()

# Export plot and data
fd.export_simulation_plot(filename)
fd.export_data(filename)
