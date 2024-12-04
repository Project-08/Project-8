from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from project8.neural_network import trainer
from project8.neural_network.differential_operators import *
import torch
import matplotlib.pyplot as plt
from project8.neural_network.implementations import loss_functions as lf


device = utils.get_device()
coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]], device)
hyperparams: utils.Params = {
    'device': device,
    # model params
    'model_constructor': 'rectangular_fnn',
    'act_fn': modules.PolyReLU(3),
    'n_in': 2,
    'n_out': 1,
    'width': 64,
    'depth': 9,
    'weight_init_fn': torch.nn.init.xavier_uniform_,
    'bias_init_fn': torch.nn.init.zeros_,
    'weight_init_kwargs': {'gain': 0.5},
    'bias_init_kwargs': {},
    # training params
    'optimizer': torch.optim.Adam,
    'lr': 1e-3,
    'n_epochs': 5000,
    'loss_fn_batch_sizes': [1000, 500],
    # problem definition
    'loss_fns': [lf.PINN.nd_laplacian, lf.dirichlet_bc_penalty],
    'loss_weights': [1, 10],
    'loss_fn_data': [
        coord_space.rand(100000), coord_space.bndry_rand(100000)
    ],
}

def main() -> None:
    model = models.NN.from_param_dict(hyperparams)
    trnr = trainer.trainer(model, hyperparams, verbose=True)
    trnr.train()
    plt.semilogy(trnr.loss_history)
    plt.show()
    pass


if __name__ == '__main__':
    main()
