from project8.neural_network import utils, models, trainer
from project8.neural_network.implementations import param_dicts, loss_functions


def main() -> None:
    # define problem
    hyperparams = param_dicts.Problem3('pinn')
    exact_sol = loss_functions.Problem3.exact
    # train model
    model = models.NN.from_param_dict(hyperparams)
    trnr = trainer.trainer(model, hyperparams, verbose=True)
    trnr.train()
    # plot output
    model.eval()
    coord_space = utils.ParameterSpace.from_rand_data(
        hyperparams['loss_fn_data'][0])
    if hyperparams['n_in'] == 2 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(200)
        output = model(grid)
        x, y = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.plot_2d(x, y, f, title='output_prediction', fig_id=2)
        if exact_sol is not None:
            real = exact_sol(grid)
            err = (output - real).abs().max()
            print(f'max error: {err}')
            u = coord_space.regrid(real)[0]
            utils.plot_2d(x, y, u, title='output_exact', fig_id=3)
    elif hyperparams['n_in'] == 3 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(50)
        grid.requires_grad = False
        output = model(grid)
        x, y, t = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.anim_2d(x, y, t, f, title='output_prediction', fig_id=2)
        if exact_sol is not None:
            real = exact_sol(grid)
            err = (output - real).abs().max()
            print(f'max error: {err}')
            u = coord_space.regrid(real)[0]
            utils.anim_2d(x, y, t, u, title='output_exact', fig_id=3)


if __name__ == '__main__':
    main()
