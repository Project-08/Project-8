from project8.neural_network import utils, models, trainer
from project8.neural_network.implementations import param_dicts, loss_functions


def main() -> None:
    problem = 1 # set to 0 to ignore
    hyperparams = param_dicts.pinn_problem1()
    model = models.NN.from_param_dict(hyperparams)
    trnr = trainer.trainer(model, hyperparams, verbose=True)
    trnr.train()
    # plot output
    coord_space = utils.ParameterSpace.from_rand_data(
        hyperparams['loss_fn_data'][0])
    if hyperparams['n_in'] == 2 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(200)
        output = model(grid)
        x, y = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.plot_2d(x, y, f, title='output_param_based', fig_id=2)
    elif problem == 1:
        grid = coord_space.fgrid(50)
        grid.requires_grad = False
        output = model(grid)
        real = loss_functions.problem_1_exact(grid)
        err = (output - real).abs().max()
        print(f'max error: {err}')
        x, y, t = coord_space.regrid(grid)
        u = coord_space.regrid(output)[0]
        f = coord_space.regrid(real)[0]
        utils.anim_2d(x, y, t, u, title='output_prediction', fig_id=2)
        utils.anim_2d(x, y, t, f, title='output_exact', fig_id=3)
    elif hyperparams['n_in'] == 3 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(50)
        output = model(grid)
        x, y, t = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.anim_2d(x, y, t, f, title='output_param_based', fig_id=2)


if __name__ == '__main__':
    main()
