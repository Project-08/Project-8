from project8.neural_network import utils, models, trainer, differential_operators as diffop, checkpoints
from project8.neural_network.implementations import param_dicts, loss_functions
import torch

def error_stats(error: torch.Tensor) -> None:
    abs_err = error.abs()
    print(f'Error stats:\n'
          f'max: {abs_err.max()}\n'
          f'min: {abs_err.abs().min()}\n'
          f'avg: {abs_err.abs().mean()}\n'
          f'std: {abs_err.abs().std()}\n'
          f'rmse: {abs_err.pow(2).mean().sqrt()}')

def main() -> None:
    # define problem
    new_model = False
    if not new_model:
        utils.set_device_override('cpu')
    hyperparams = param_dicts.Problem2('drm') # 'pinn' or 'drm'
    exact_sol = loss_functions.Problem2.exact
    # define model
    # Make a new one
    if new_model:
        model = models.NN.from_param_dict(hyperparams)
        trnr = trainer.trainer(model, hyperparams, verbose=True)
        trnr.train()
    # Or load a saved one
    else:
        model = checkpoints.load_model('problem2_drm.pth')
        model.to(hyperparams['device'])

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
            err = (output - real)
            error_stats(err)
            u = coord_space.regrid(real)[0]
            err = coord_space.regrid(err)[0]
            utils.plot_2d(x, y, err, title='output_error', fig_id=3)
            utils.plot_2d(x, y, u, title='output_exact', fig_id=4)
    elif hyperparams['n_in'] == 3 and hyperparams['n_out'] == 1:
        grid = coord_space.fgrid(50)
        grid.requires_grad = False
        output = model(grid)
        x, y, t = coord_space.regrid(grid)
        f = coord_space.regrid(output)[0]
        utils.anim_2d(x, y, t, f, title='output_prediction', fig_id=2)
        if exact_sol is not None:
            real = exact_sol(grid)
            err = (output - real)
            error_stats(err)
            u = coord_space.regrid(real)[0]
            err = coord_space.regrid(err)[0]
            utils.anim_2d(x, y, t, err, title='output_error', fig_id=3)
            utils.anim_2d(x, y, t, u, title='output_exact', fig_id=4)
    elif hyperparams['n_in'] == 3 and hyperparams['n_out'] > 1:
        grid = coord_space.fgrid(50)
        grid.requires_grad = False
        model(grid)
        diffop.add_output_col(model, (model.output[:, 0].pow(2) + model.output[:, 1].pow(2)).sqrt().unsqueeze(1))
        output = model.output
        x, y, t = coord_space.regrid(grid)
        if exact_sol is not None:
            real = exact_sol(grid)
        for i in range(output.shape[1]):
            f = coord_space.regrid(output[:, i].unsqueeze(1))[0]
            utils.anim_2d(x, y, t, f, title=f'output_{i}_prediction', fig_id=2*i)
            if exact_sol is not None:
                err = (output - real[:, i]).abs().max()
                print(f'output {i}')
                error_stats(err)
                u = coord_space.regrid(real[:, i].unsqueeze(1))[0]
                utils.anim_2d(x, y, t, u, title=f'output_{i}_exact', fig_id=2*i+1)

    if new_model and (name := input('Enter name to save model: ')) != '':
        checkpoints.save_model_state(model, name)




if __name__ == '__main__':
    main()
