from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from project8.neural_network import trainer
import torch
import matplotlib.pyplot as plt
from project8.neural_network.implementations import param_dicts


def main() -> None:
    hyperparams = param_dicts.pinn_2d_laplacian()
    model = models.NN.from_param_dict(hyperparams)
    trnr = trainer.trainer(model, hyperparams, verbose=True)
    trnr.train()
    coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]],
                                       hyperparams['device'])
    # plot output
    grid = coord_space.fgrid(200)
    output = model(grid)
    x, y = coord_space.regrid(grid)
    f = coord_space.regrid(output)[0]
    utils.plot_2d(x, y, f, title='output_param_based', fig_id=2)
    pass


if __name__ == '__main__':
    main()
