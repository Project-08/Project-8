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
    def random_search(hyperparams, n_points=10, top_k=5, write_results=True, name="random_search_results.txt"):
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
                f"Testing hyperparameters {len(results) + 1}:")  # Progress feedback
            for key in varying_hps:
                print(f"{key}: {params[key]}")
            try:
                model, loss = GridSearch.grid_point(params)
            except Exception as e:
                print(f"Error with params {params}: {e}")
                continue
            print(f"Loss: {loss}")
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
            GridSearch.write_search_results(top_k, varying_hps, results, name)
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


# Early termination
# after 1500 epoch with no improvement

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

def pinn_2d_search(param_dict, name="pinn_2d_search_results.txt"):
    hyperparameters = GridSearch.listify(param_dict)
    hyperparameters['n_epochs'] = [3000]
    hyperparameters['width'] = [16, 32, 64, 128]
    hyperparameters['depth'] = [2, 4, 8, 10]
    hyperparameters['act_fn'] = [
        modules.PolyReLU(2),
        modules.PolyReLU(3),
        modules.PolyReLU(4),
        modules.Sin(),
        nn.Tanh(),
        nn.SiLU()
    ]
    hyperparameters['lr'] = [1e-2, 1e-3, 1e-4, 1e-5]
    hyperparameters['weight_init_kwargs'] = [
        {'gain': 0.5},
        {'gain': 1.0},
        {'gain': 1.5},
    ]

    print(
        f"Number of combinations: "
        f"{GridSearch.n_combinations(hyperparameters)}")
    GridSearch.random_search(
        hyperparameters,
        top_k=50,
        n_points=50,
        write_results=True,
        name=name
    )

def drm_2d_search(param_dict, name="drm_2d_search_results.txt"):
    hyperparameters = GridSearch.listify(param_dict)
    hyperparameters['n_epochs'] = [3000]
    hyperparameters['width'] = [16, 32, 64, 128]
    hyperparameters['n_blocks'] = [1, 2, 4]
    hyperparameters['n_linear_drm'] = [2, 3]
    hyperparameters['act_fn'] = [
        nn.ReLU(),
        modules.PolyReLU(2),
        modules.PolyReLU(3),
        modules.PolyReLU(4),
        nn.Tanh(),
        nn.SiLU()
    ]
    hyperparameters['lr'] = [1e-2, 1e-3, 1e-4, 1e-5]
    hyperparameters['weight_init_kwargs'] = [
        {'gain': 0.5},
        {'gain': 1.0},
        {'gain': 1.5},
    ]

    print(
        f"Number of combinations: "
        f"{GridSearch.n_combinations(hyperparameters)}")
    GridSearch.random_search(
        hyperparameters,
        top_k=50,
        n_points=50,
        write_results=True,
        name=name
    )

def pinn_3d_search(param_dict, name="pinn_3d_search_results.txt"):
    hyperparameters = GridSearch.listify(param_dict)
    hyperparameters['n_epochs'] = [3000]
    hyperparameters['width'] = [32, 64, 128, 256]
    hyperparameters['depth'] = [2, 4, 8, 10]
    hyperparameters['act_fn'] = [
        modules.PolyReLU(2),
        modules.PolyReLU(3),
        modules.PolyReLU(4),
        modules.Sin(),
        nn.Tanh(),
        nn.SiLU()
    ]
    hyperparameters['lr'] = [1e-2, 1e-3, 1e-4, 1e-5]
    hyperparameters['weight_init_kwargs'] = [
        {'gain': 0.5},
        {'gain': 1.0},
        {'gain': 1.5},
    ]

    print(
        f"Number of combinations: "
        f"{GridSearch.n_combinations(hyperparameters)}")
    GridSearch.random_search(
        hyperparameters,
        top_k=50,
        n_points=50,
        write_results=True,
        name=name
    )

def drm_3d_search(param_dict, name="drm_3d_search_results.txt"):
    hyperparameters = GridSearch.listify(param_dict)
    hyperparameters['n_epochs'] = [3000]
    hyperparameters['width'] = [32, 64, 128, 256]
    hyperparameters['n_blocks'] = [1, 2, 4]
    hyperparameters['n_linear_drm'] = [2, 3]
    hyperparameters['act_fn'] = [
        nn.ReLU(),
        modules.PolyReLU(2),
        modules.PolyReLU(3),
        modules.PolyReLU(4),
        nn.Tanh(),
        nn.SiLU()
    ]
    hyperparameters['lr'] = [1e-2, 1e-3, 1e-4, 1e-5]
    hyperparameters['weight_init_kwargs'] = [
        {'gain': 0.5},
        {'gain': 1.0},
        {'gain': 1.5},
    ]

    print(
        f"Number of combinations: "
        f"{GridSearch.n_combinations(hyperparameters)}")
    GridSearch.random_search(
        hyperparameters,
        top_k=50,
        n_points=50,
        write_results=True,
        name=name
    )


if __name__ == "__main__":
    # running now
    # pinn_3d_search(param_dicts.Problem3('pinn'), name="p3_pinn_search_results.txt")
    # drm_3d_search(param_dicts.Problem3('drm'), name="p3_drm_search_results.txt")
    # pinn_3d_search(param_dicts.Problem4('pinn'), name="p4_pinn_search_results.txt")
    # drm_3d_search(param_dicts.Problem4('drm'), name="p4_drm_search_results.txt")

    # previously done
    # pinn_3d_search(param_dicts.Problem5('pinn'), name="p5_pinn_search_results.txt")
    # drm_3d_search(param_dicts.Problem5('drm'), name="p5_drm_search_results.txt")

    pinn_3d_search(param_dicts.Problem6('pinn'), name="p6_pinn_search_results.txt")
    drm_3d_search(param_dicts.Problem6('drm'), name="p6_drm_search_results.txt")
