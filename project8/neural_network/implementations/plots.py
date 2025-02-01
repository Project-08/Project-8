from project8.neural_network import utils, models, trainer, differential_operators as diffop, checkpoints
from project8.neural_network.implementations import param_dicts, loss_functions
import torch
import pickle


def plot_(params, exact_sol, title: str, fig_id: int):
    model = models.NN.from_param_dict(params)
    trnr = trainer.trainer(model, params, verbose=True)
    return trnr.train_plot(exact_sol, title, fig_id, max_time=30, max_epochs=10000)


if __name__ == '__main__':
    params = [
        param_dicts.Problem1('pinn'),
        param_dicts.Problem2('pinn'),
        param_dicts.Problem3('pinn'),
        param_dicts.Problem4('pinn'),
        param_dicts.Problem5('pinn'),
        param_dicts.Problem6('pinn'),
        param_dicts.Problem1('drm'),
        param_dicts.Problem2('drm'),
        param_dicts.Problem3('drm'),
        param_dicts.Problem4('drm'),
        param_dicts.Problem5('drm'),
        param_dicts.Problem6('drm'),
    ]
    exacts = [
        loss_functions.Problem1.exact,
        loss_functions.Problem2.exact,
        loss_functions.Problem3.exact,
        loss_functions.Problem4.exact,
        loss_functions.Problem5.exact,
        loss_functions.Problem6.exact,
        loss_functions.Problem1.exact,
        loss_functions.Problem2.exact,
        loss_functions.Problem3.exact,
        loss_functions.Problem4.exact,
        loss_functions.Problem5.exact,
        loss_functions.Problem6.exact,
    ]

    titles = [
        r'P1 PINN: $\Delta u = \sin(\pi x) \sin(\pi y)$',
        r'P2 PINN: $\Delta u = (x^2 + y^2) e^{xy}$',
        r'P3 PINN: $\Delta u = 2(xy + xz + yz)$',
        r'P4 PINN: $\Delta u = 6$',
        r'P5 PINN: $\Delta u = -\pi^2 xy \sin(\pi z)$',
        r'P6 PINN: $Delta u = -3\pi^2 \sin(\pi x) \sin(\pi y) \sin(\pi z)$',
        r'P1 DRM: $\Delta u = \sin(\pi x) \sin(\pi y)$',
        r'P2 DRM: $\Delta u = (x^2 + y^2) e^{xy}$',
        r'P3 DRM: $\Delta u = 2(xy + xz + yz)$',
        r'P4 DRM: $\Delta u = 6$',
        r'P5 DRM: $\Delta u = -\pi^2 xy \sin(\pi z)$',
        r'P6 DRM: $\Delta u = -3\pi^2 \sin(\pi x) \sin(\pi y) \sin(\pi z)$'
    ]

    result_lists = {}

    for i in range(len(params)):
        key = titles[i].split(':')[0]
        t, err = plot_(params[i], exacts[i], titles[i],  i)
        result_lists[key] = (t, err)

    with open('error_v_time_results.pkl', 'wb') as f:
        pickle.dump(result_lists, f)