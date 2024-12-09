from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from project8.neural_network import trainer
import torch
import matplotlib.pyplot as plt
from project8.neural_network.implementations import param_dicts



def main() -> None:
    hyperparams = param_dicts.pinn_2d_wave()
    coord_space = utils.ParameterSpace.from_rand_data(
        hyperparams['loss_fn_data'][1])
    print(coord_space.domain)
    model = models.NN.from_param_dict(hyperparams)
    trnr = trainer.trainer(model, hyperparams, verbose=True)
    trnr.train()
    # plot output
    if hyperparams['n_in'] == 2 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(200)
        output = model(grid)
        x, y = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.plot_2d(x, y, f, title='output_param_based', fig_id=2)
    elif hyperparams['n_in'] == 3 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(50)
        output = model(grid)
        x, y, t = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.anim_2d(x, y, t, f, title='output_param_based', fig_id=2)


if __name__ == '__main__':
    main()
