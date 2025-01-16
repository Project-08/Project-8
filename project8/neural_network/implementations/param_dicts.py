from project8.neural_network import utils
from project8.neural_network import modules
import torch
from project8.neural_network.implementations import loss_functions as lf


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
            lf.PINN.laplacian(lf.sinxy_source),
            lf.General.dirichlet_bc
        ],
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
        'loss_fns': [
            lf.DRM.laplacian(lf.sinxy_source),
            lf.General.dirichlet_bc
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
        'act_fn': torch.nn.ReLU(),
        'n_in': 3,
        'n_out': 1,
        'width': 256,
        'depth': 6,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 1},
        'bias_init_kwargs': {},
        # training params
        'optimizer': torch.optim.AdamW,
        'lr': 1e-4,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 400],
        # problem definition
        'loss_fns': [
            lf.PINN.wave(lf.nmfde_a4_wave),
            lf.General.dirichlet_bc],
        'loss_weights': [100, 1],
        'loss_fn_data': [
            coord_space.rand(1_000_000),
            coord_space.select_bndry_rand(100_000, boundary_select)
        ],
    }
    return hyperparams


def pinn_problem5() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 1]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 1], [0, 0]], device=device)
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
        'loss_fn_batch_sizes': [1000, 400, 200],
        # problem definition
        'loss_fns': [
            lf.PINN.laplacian(lf.Problem5.source),
            lf.General.dirichlet_bc,
            lf.Problem5.bc,
        ],
        'loss_weights': [1, 100, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
        ],
    }
    return hyperparams


def drm_problem5() -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 1]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 1], [0, 0]], device=device)
    hyperparams: utils.Params = {
        'device': device,
        # model params
        'model_constructor': 'drm',
        'act_fn': modules.PolyReLU(3),
        'n_in': 3,
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
        'loss_fn_batch_sizes': [1000, 400, 200],
        # problem definition
        'loss_fns': [
            lf.DRM.laplacian(lf.Problem5.source),
            lf.General.dirichlet_bc,
            lf.Problem5.bc,
        ],
        'loss_weights': [1, 100, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
        ],
    }
    return hyperparams


def Problem1(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 2,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [500, 300],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 20,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem1.source),
                    lf.General.dirichlet_bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 20,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem1.source),
                    lf.General.dirichlet_bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams


def Problem2(method: str = 'pinn',
             domain: tuple[float | int, float | int] = (0, 1)) -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([domain, domain], device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 2,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [500, 300],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 20,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem2.source),
                    lf.Problem2.bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 20,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem2.source),
                    lf.Problem2.bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams


def Problem3(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 0]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 1], [0, 1]], device=device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 3,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 300, 300],
        'loss_weights': [1, 100, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 256,
                'depth': 8,
                'loss_weights': [1, 10, 100],
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem3.source),
                    lf.General.dirichlet_bc,
                    lf.Problem3.bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 64,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem3.source),
                    lf.General.dirichlet_bc,
                    lf.Problem3.bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams


def Problem4(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 3,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 600],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 256,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem4.source),
                    lf.Problem4.bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 64,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem4.source),
                    lf.Problem4.bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams


def Problem5(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    bc_select_1 = torch.tensor([[1, 0], [1, 0], [1, 1]], device=device)
    bc_select_2 = torch.tensor([[0, 1], [0, 1], [0, 0]], device=device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 3,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 400, 200],
        'loss_weights': [1, 100, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 256,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem5.source),
                    lf.General.dirichlet_bc,
                    lf.Problem5.bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 64,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem5.source),
                    lf.General.dirichlet_bc,
                    lf.Problem5.bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams


def Problem6(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1], [0, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 3,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 600],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 256,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem6.source),
                    lf.General.dirichlet_bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 64,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'loss_fns': [
                    lf.DRM.laplacian(lf.Problem6.source),
                    lf.General.dirichlet_bc,
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams
