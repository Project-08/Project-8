from project8.neural_network import utils, models, trainer, differential_operators as diffop, checkpoints
from project8.neural_network.implementations import param_dicts, loss_functions
import torch

"""
new_model = True to train a new model
Training happens on gpu unless not available

new_model = False to load a model
select which model to load by changing the name in 
checkpoints.load_model('name.pth').
The models are saved in project8/models.
Inferring happens on cpu by default so there is more memory available
you can change infer device by setting 
utils.set_device_override to 'cuda' or 'cpu'

hyperparams = param_dicts.Problem(1-6)('pinn') or ('drm')
Select which set of hyperparameters and problem definition to use

exact_sol = loss_functions.Problem(1-6).exact
Select the exact solution to compare the output with,
 if None then no comparison is done
"""


def main() -> None:
    # define problem
    new_model = False       # Set to True to train a new model
    if not new_model:
        utils.set_device_override('cpu')        # Default to infer is on cpu
    hyperparams = param_dicts.Problem5('pinn') # Problem(1-6) 'pinn' or 'drm'
    exact_sol = loss_functions.Problem5.exact # Problem(1-6).exact, make sure to change this with the problem
    # define model
    if new_model:
        model = models.NN.from_param_dict(hyperparams)
        trnr = trainer.trainer(model, hyperparams, verbose=True)
        trnr.train()
    else: # change the model to load here, they are saved in the models folder
        model = checkpoints.load_model('p5_pinn_opt.pth')
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
            utils.error_stats(err)
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
            utils.error_stats(err)
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
                utils.error_stats(err)
                u = coord_space.regrid(real[:, i].unsqueeze(1))[0]
                utils.anim_2d(x, y, t, u, title=f'output_{i}_exact', fig_id=2*i+1)

    if new_model and (name := input('Enter name to save model: ')) != '':
        checkpoints.save_model_state(model, name)




if __name__ == '__main__':
    main()
