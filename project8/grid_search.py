import itertools
import torch
from torch.optim.lr_scheduler import MultiStepLR
from copy import deepcopy
import sys
import os
# Append the directory containing DeepRitz.py to the sys.path list
# This line assumes your script is run from within the project8/ directory
sys.path.append(os.path.abspath('old'))

import DeepRitz as dr
import areaVolume as av

def grid_search(grid_params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    best_error = float('inf')
    best_params = None

    # Generate all combinations of parameters
    keys, values = zip(*grid_params.items())
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    for params in param_combinations:
        print(f"Testing parameters: {params}")
        model = dr.RitzNet(params).to(device)
        model.apply(dr.initWeights)

        optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'], weight_decay=params.get('decay', 0.0001))
        scheduler = MultiStepLR(optimizer, milestones=params.get('milestone', [10000, 20000]), gamma=params.get('gamma', 0.5))

        # Adjust params to include necessary elements missing from grid that are used in training
        params['device'] = device
        dr.preTrain(model, device, params, optimizer, scheduler, dr.rough)  # Assuming rough is defined elsewhere
        dr.train(model, device, params, optimizer, scheduler)

        model.eval()
        test_error = dr.test(model, device, params)

        if test_error < best_error:
            best_error = test_error
            best_params = params.copy()

        print(f"Parameters: {params}, Test Error: {test_error}")

    return best_params, best_error

fixed_params = {
    "radius": [1],         # Physical or geometrical parameter, likely fixed by the problem's nature
    "d": [10],             # Dimensionality of the input space
    "dd": [1],             # Dimensionality of the output space
    "preLr": [0.01],       # Pre-training learning rate (you might tie this to 'lr' if they are always the same)
    "numQuad": [40000],    # Number of quadrature points for testing, specific to your simulation or problem
    "preStep": [0],        # Pre-training steps, useful if you have a pre-training phase
    "diff": [0.001],       # Differential increment for numerical calculations
    "writeStep": [50],     # How often to write out data or logs
    "sampleStep": [10],    # Sampling frequency for dynamically adjusting training samples
    "area": None,  # Calculate this once if it doesn't change
    "step_size": [5000],   # Step size for learning rate scheduler, could be varied if using a scheduler in grid search
}
fixed_params['area'] = [av.areaVolume(fixed_params['radius'][0], fixed_params['d'][0])]

variable_params = {
    'lr': [0.01, 0.005, 0.001],
    'preLr': None,
    'depth': [3, 4, 5],
    'width': [10, 20, 30],
    'decay': [0.0001, 0.0005, 0.001],
    'penalty': [100, 500, 1000],
    'bodyBatch': [512, 1024, 2048],
    'bdryBatch': [1024, 2048, 4096],
    'gamma': [0.5, 0.7, 0.9]
}
variable_params['preLr'] = variable_params['lr']

def configure_params(fixed_params, variable_params):
    # Merge fixed and variable parameters for a complete configuration
    params = deepcopy(fixed_params)
    params.update(variable_params)  # Update with the variable parameters specific for this configuration
    return params

if __name__ == "__main__":
    parameter_grid = configure_params(fixed_params, variable_params)
    best_params, best_error = grid_search(parameter_grid)
    print(f"Best Parameters: {best_params}")
    print(f"Lowest Test Error: {best_error}")
