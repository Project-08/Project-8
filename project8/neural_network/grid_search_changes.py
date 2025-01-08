import itertools
from project8.neural_network import models, trainer, utils, modules
from project8.neural_network.implementations import param_dicts
import torch.nn as nn
from copy import deepcopy
import random
from warnings import warn


class GridSearch:
    @staticmethod
    def grid_point(hyperparams):
        model = models.NN.from_param_dict(hyperparams)
        trnr = trainer.trainer(model, hyperparams)
        trnr.train()
        return model, trnr.best_loss

    @staticmethod
    def grid_search(hyperparams, top_k=5, write_results=True):
        varying_hps = [key for key in hyperparams if len(hyperparams[key]) > 1]
        best_model_state = None
        best_loss = float('inf')
        best_params = None
        keys, values = zip(*hyperparams.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        results = []
        for params in combinations:
            print(
                f"Testing hyperparameters:")  # Progress feedback
            for key in varying_hps:
                print(f"{key}: {params[key]}")
            try:
                model, loss = GridSearch.grid_point(params)
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
            results.append((params, loss, deepcopy(model.state_dict())))
        if not results:
            print("No valid results found.")
            return None, None, None
        results.sort(key=lambda x: x[1])
        # Print top_k results
        print(f"\nTop {top_k} results (sorted by loss):")
        for i, (params, loss, _) in enumerate(results[:top_k]):
            print(f"Rank {i+1}, loss = {loss}")
            for key in varying_hps:
                print(f"    {key}: {params[key]}")

        # write the results to a file, if it doesnt exist, create it
        if write_results:
            GridSearch.write_search_results(top_k, varying_hps, results, "grid_search_results.txt")
        best_params, best_loss, best_model_state = results[0]
        return best_params, best_loss, best_model_state

    @staticmethod
    def write_search_results(top_k, varying_hps, results, file_name, overwrite=False):
        mode = "w" if overwrite else "a"
        with open(file_name, mode) as f: # if we want to overwrite, change to "w"
            f.write(f"\nTop {top_k} results (sorted by loss):\n")
            for i, (params, loss, _) in enumerate(results[:top_k]):
                f.write(f"Rank {i+1}, loss = {loss}\n")
                for key in varying_hps:
                    f.write(f"    {key}: {params[key]}\n")
            # a line to separate different searches
            f.write("\n" + "-"*50 + "\n")

    @staticmethod
    def random_search(hyperparams, n_points=10, top_k=5, write_results=True):
        n_combs = GridSearch.n_combinations(hyperparams)
        if n_points > n_combs:
            warn(
                f"Number of points to search exceeds number of combinations: "
                f"{n_points} > {n_combs}. Searching all...")
            n_points = n_combs
        varying_hps = [key for key in hyperparams if len(hyperparams[key]) > 1]
        best_model_state = None
        best_loss = float('inf')
        best_params = None
        keys, values = zip(*hyperparams.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        # search n_points random combinations
        random.shuffle(combinations)
        results = []
        for params in combinations[:n_points]:
            print(
                f"Testing hyperparameters:")  # Progress feedback
            for key in varying_hps:
                print(f"{key}: {params[key]}")
            try:
                model, loss = GridSearch.grid_point(params)
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
            results.append((params, loss, deepcopy(model.state_dict())))
        if not results:
            print("No valid results found.")
            return None, None, None
        
        results.sort(key=lambda x: x[1])
        # Print top_k results
        print(f"\nTop {top_k} results (sorted by loss):")
        for i, (params, loss, _) in enumerate(results[:top_k]):
            print(f"Rank {i+1}, loss = {loss}")
            for key in varying_hps:
                print(f"    {key}: {params[key]}")

        if write_results:
            GridSearch.write_search_results(top_k, varying_hps, results, "random_search_results.txt")
        best_params, best_loss, best_model_state = results[0]
        return best_params, best_loss, best_model_state

    @staticmethod
    def listify(hyperparams):
        for key in hyperparams:
            hyperparams[key] = [hyperparams[key]]
        return hyperparams

    @staticmethod
    def n_combinations(hyperparams):
        n = 1
        for key in hyperparams:
            n *= len(hyperparams[key])
        return n


# this file works on the nn improvements branch
# didnt push it because i havent figured out type hints for it yet

# Early termination
# on which conditions? could be implemenmted in trainer class

# best loss: last epoch loss?
# now best loss from trainer object, which is best loss during training

# Do we store best loss model parameters?
# we now store the state_dict of the model which can be used with
# params to recreate the model

# We probably want to return the 5-10 best models parameters and their
# losses to compare the impact of hyperparameters

# we could randmize the order of the combinations to test and set a number
# of points to test to randomly select a subset of the combinations
# see random search

# maybe try to implement bayesian optimization,
# that seems (to) complicated though
# https://botorch.org/
# https://arxiv.org/pdf/1807.02811
# or other optimization techniques

# allow continuous hyperparameters? maybe unnecessary

# Example usage
def main():
    hyperparameters = param_dicts.drm_2d_laplacian()
    hyperparameters = GridSearch.listify(hyperparameters)
    # now all hyperparameters are lists with 1 entry,
    # extend the ones you want to vary
    hyperparameters['n_epochs'] = [2000]
    hyperparameters['loss_fn_batch_sizes'] = [[100, 50]]
    # Search domains for key hyperparameters
    hyperparameters['width'] = [16, 32, 64, 128]  # Increasing capacity
    hyperparameters['n_blocks'] = [3, 5, 8, 10]  # Experimenting with depth
    hyperparameters['act_fn'] = [
        modules.PolyReLU(2),
        modules.PolyReLU(3),
        modules.PolyReLU(4),
        nn.Tanh(),            # Smooth activation
        nn.ReLU(),            # Standard activation
        nn.SiLU()             # Swish-like activation
    ]
    hyperparameters['lr'] = [1e-2, 1e-3, 1e-4, 1e-5]  # Covering a wide range of learning rates

    print(
        f"Number of combinations: "
        f"{GridSearch.n_combinations(hyperparameters)}")
    best_params, best_loss, best_state = GridSearch.grid_search(
        hyperparameters,top_k=100)


# results of current settings
"""Best Hyperparameters: {'device': 'cuda:0', 'model_constructor': 'drm', 
'act_fn': ReLU(), 'n_in': 2, 'n_out': 1, 'width': 64, 'n_blocks': 5, 
'n_linear_drm': 2, 'weight_init_fn': <function xavier_uniform_ at 
0x7cc8723aa660>, 'bias_init_fn': <function zeros_ at 0x7cc8723aa3e0>, 
'weight_init_kwargs': {'gain': 0.5}, 'bias_init_kwargs': {}, 'optimizer': 
<class 'torch.optim.adam.Adam'>, 'lr': 0.0001, 'n_epochs': 1000, 
'loss_fn_batch_sizes': [100, 50], 'loss_fns': [
<project8.neural_network.implementations.loss_functions.DRM.nd_laplacian 
object at 0x7cc85b187650>, 
<project8.neural_network.implementations.loss_functions.General
.dirichlet_bc_penalty object at 0x7cc85b187710>], 'loss_weights': [1, 10], 
'loss_fn_data': [tensor([[-0.2189, -0.6418],
        [-0.8135,  0.8704],
        [ 0.4982, -0.2455],
        ...,
        [ 0.1399,  0.9482],
        [ 1.0056, -0.4889],
        [-0.9005, -0.8499]], device='cuda:0', dtype=torch.float64), tensor([
        [-1.0000,  0.1905],
        [ 1.1484, -1.0000],
        [-1.0000, -0.0922],
        ...,
        [-1.0000, -0.7840],
        [ 0.0703, -1.0000],
        [ 1.6922,  1.0000]], device='cuda:0', dtype=torch.float64)]}
Best Loss: -0.004653674550354481"""

if __name__ == "__main__":
    main()
