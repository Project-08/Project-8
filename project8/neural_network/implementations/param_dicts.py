from project8.neural_network import utils
from project8.neural_network import modules
import torch
from project8.neural_network.implementations import loss_functions as lf


def default() -> utils.Params:
    hyperparams: utils.Params = {
        'device': 'cpu',
        # model params
        'model_constructor': 'rectangular_fnn',
        'act_fn': modules.PolyReLU(3),
        'n_in': 1,
        'n_out': 1,
        'width': 32,
        'depth': 7,
        'n_blocks': 4,
        'n_linear_drm': 2,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        # training params
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [],
        # problem definition
        'loss_fns': [],
        'loss_weights': [],
        'loss_fn_data': []
    }
    return hyperparams


def pinn_2d_laplacian() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        # model params
        'model_constructor': 'rectangular_fnn',
        'act_fn': modules.FourierLike(64),
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
        'loss_fn_batch_sizes': [100, 50],
        # problem definition
        'loss_fns': [lf.PINN.nd_laplacian, lf.dirichlet_bc_penalty],
        'loss_weights': [1, 10],
        'loss_fn_data': [
            coord_space.rand(100000), coord_space.bndry_rand(100000)
        ],
    }
    return hyperparams


def drm_2d_laplacian() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        # model params
        'model_constructor': 'drm',
        'act_fn': modules.PolyReLU(3),
        'n_in': 2,
        'n_out': 1,
        'width': 20,
        'n_blocks': 4,
        'n_linear_drm': 2,
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
        'loss_fns': [lf.DRM.nd_laplacian, lf.dirichlet_bc_penalty],
        'loss_weights': [1, 10],
        'loss_fn_data': [
            coord_space.rand(100000), coord_space.bndry_rand(100000)
        ],
    }
    return hyperparams


def pinn_2d_wave() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 10], [0, 5], [0, 4]], device)
    boundary_select = torch.tensor(
        [[1, 1], [1, 1], [1, 0]], device=device, dtype=torch.float64)
    hyperparams: utils.Params = {
        'device': device,
        # model params
        'model_constructor': 'rectangular_fnn',
        'act_fn': torch.nn.Tanh(),
        'n_in': 3,
        'n_out': 1,
        'width': 64,
        'depth': 7,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        # training params
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 800],
        # problem definition
        'loss_fns': [lf.PINN.wave_2d, lf.dirichlet_bc_penalty],
        'loss_weights': [1, 10],
        'loss_fn_data': [
            coord_space.rand(1000000),
            coord_space.select_bndry_rand(1000000, boundary_select)
        ],
    }
    return hyperparams
