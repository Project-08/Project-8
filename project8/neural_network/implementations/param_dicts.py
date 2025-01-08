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
        'loss_fns': [
            lf.PINN.nd_laplacian(lf.sinxy_source),
            lf.General.dirichlet_bc_penalty()
        ],
        'loss_weights': [1, 10],
        'loss_fn_data': [
            coord_space.rand(100000), coord_space.bndry_rand(100000)
        ],
    }
    return hyperparams


def nmfde_a3() -> utils.Params:
    hyperparams = default()
    hyperparams['device'] = utils.get_device()
    hyperparams['n_epochs'] = 5000
    hyperparams['model_constructor'] = 'rectangular_fnn'
    hyperparams['n_in'] = 2
    hyperparams['n_out'] = 1
    hyperparams['width'] = 64
    hyperparams['depth'] = 9
    hyperparams['act_fn'] = modules.FourierLike(64)
    coord_space = utils.ParameterSpace([[0, 10], [0, 5]],
                                       hyperparams['device'])
    hyperparams['loss_fns'] = [
        lf.PINN.nd_laplacian(lf.nmfde_a3_source),
        lf.General.dirichlet_bc_penalty()
    ]
    hyperparams['loss_weights'] = [1, 10]
    hyperparams['loss_fn_data'] = [
        coord_space.rand(100000), coord_space.bndry_rand(100000)
    ]
    hyperparams['loss_fn_batch_sizes'] = [300, 100]
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
        'loss_fns': [
            lf.DRM.nd_laplacian(lf.sinxy_source),
            lf.General.dirichlet_bc_penalty()
        ],
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
        'act_fn': torch.nn.Sigmoid(),
        'n_in': 3,
        'n_out': 1,
        'width': 1024,
        'depth': 3,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 1},
        'bias_init_kwargs': {},
        # training params
        'optimizer': torch.optim.AdamW,
        'lr': 1e-5,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [2000, 600],
        # problem definition
        'loss_fns': [
            lf.PINN.nd_wave(lf.nmfde_a4_wave),
            lf.General.dirichlet_bc_penalty()],
        'loss_weights': [1, 10],
        'loss_fn_data': [
            coord_space.rand(10_000_000),
            coord_space.select_bndry_rand(1_000_000, boundary_select)
        ],
    }
    return hyperparams

def pinn_problem1() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 1]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 0], [0, 0]], device=device)
    bc_select_3 = torch.tensor([[0, 0], [0, 1], [0, 0]], device=device)
    hyperparams: utils.Params = {
        'device': device,
        # model params
        'model_constructor': 'rectangular_fnn',
        'act_fn': torch.nn.Tanh(),
        'n_in': 3,
        'n_out': 1,
        'width': 256,
        'depth': 8,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        # training params
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [100, 50, 20, 20],
        # problem definition
        'loss_fns': [
            lf.PINN.nd_laplacian(lf.problem1_source),
            lf.General.dirichlet_bc_penalty(),
            lf.General.problem1_bc_x(),
            lf.General.problem1_bc_y()
        ],
        'loss_weights': [1, 1, 1, 1],
        'loss_fn_data': [
            coord_space.rand(1_000_000),
            coord_space.select_bndry_rand(400_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
            coord_space.select_bndry_rand(100_000, bc_select_3)
        ],
    }
    return hyperparams

def drm_problem1() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 1]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 0], [0, 0]], device=device)
    bc_select_3 = torch.tensor([[0, 0], [0, 1], [0, 0]], device=device)
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
        'loss_fn_batch_sizes': [100, 100, 100, 100],
        # problem definition
        'loss_fns': [
            lf.DRM.nd_laplacian(lf.problem1_source),
            lf.General.dirichlet_bc_penalty(),
            lf.General.problem1_bc_x(),
            lf.General.problem1_bc_y()
        ],
        'loss_weights': [1, 1, 1, 1],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
            coord_space.select_bndry_rand(100_000, bc_select_3)
        ],
    }
    return hyperparams