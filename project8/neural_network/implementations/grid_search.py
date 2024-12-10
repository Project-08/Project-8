import itertools
from project8.neural_network.implementations.drm_2d_poisson_class import DRM2DPoisson
from project8.neural_network.implementations.pinn_2d_poisson_class import PINN2DPoisson
from project8.neural_network.implementations.pinn_2d_wave_class import PINN2DWave

class GridSearch:
    @staticmethod
    def grid_search(hyperparams, model_factories):
        best_model = None
        best_loss = float('inf')
        best_params = None
        keys, values = zip(*hyperparams.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
        
        for model_factory in model_factories:
            for params in combinations:
                print(f"Testing hyperparameters: {params}")  # Progress feedback
                try:
                    model = model_factory(**params)
                    loss = model.train()
                except Exception as e:
                    print(f"Error with params {params}: {e}")
                    continue
                
                if loss < best_loss:
                    best_loss = loss
                    best_model = model
                    best_params = params
                    
        return best_model, best_params, best_loss

# Example usage  
def main():
    hyperparameters = {
        'n_inputs': [3],
        'n_outputs': [1],
        # 'width': [20],
        # 'n_blocks': [4],
        'lr': [1e-3],
        'n_epochs': [5000],
        # 'loss_weights': [(1, 100), (1, 50)],
    }
    # Only the specified hyperparameters will be varied
    best_model, best_params, best_loss = GridSearch.grid_search(hyperparameters, [PINN2DWave])

    print("\nBest Model:", best_model)
    print("Best Hyperparameters:", best_params)
    print("Best Loss:", best_loss)

    best_model.evaluate()

if __name__ == "__main__":
    main()