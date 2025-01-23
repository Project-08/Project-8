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


def Problem1(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    coord_space = utils.ParameterSpace([[0, 1], [0, 1]], device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 2,
        'n_out': 1,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 600],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn': # optimized
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'weight_init_kwargs': {'gain': 1.5},
                'act_fn': torch.nn.SiLU(),
                'width': 64,
                'depth': 10,
                'lr': 1e-2,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem1.source),
                    lf.General.dirichlet_bc,
                ],
            })
        case 'drm': # optimized
            hyperparams.update({
                'model_constructor': 'drm',
                'weight_init_kwargs': {'gain': 1.0},
                'act_fn': modules.PolyReLU(2),
                'width': 16,
                'n_blocks': 4,
                'n_linear_drm': 2,
                'lr': 1e-2,
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
        'weight_init_kwargs': {'gain': 1.0},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 600],
        'loss_weights': [1, 100],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.bndry_rand(100_000),
        ],
    }
    match method:
        case 'pinn': # optimized
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': modules.PolyReLU(2),
                'width': 64,
                'depth': 8,
                'lr': 1e-3,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem2.source),
                    lf.Problem2.bc,
                ],
            })
        case 'drm':
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(3),
                'width': 16,
                'n_blocks': 1,
                'n_linear_drm': 2,
                'lr': 1e-3,
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
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'n_epochs': 5000,
        'loss_fn_batch_sizes': [1000, 300, 300],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_1),
            coord_space.select_bndry_rand(100_000, bc_select_2),
        ],
    }
    match method:
        case 'pinn': # optimized
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'weight_init_kwargs': {'gain': 1.0},
                'act_fn': modules.PolyReLU(3),
                'width': 256,
                'depth': 4,
                'lr': 1e-3,
                'loss_weights': [1, 10, 100],
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem3.source),
                    lf.General.dirichlet_bc,
                    lf.Problem3.bc,
                ],
            })
        case 'drm': # optimized
            hyperparams.update({
                'model_constructor': 'drm',
                'weight_init_kwargs': {'gain': 1.5},
                'act_fn': modules.PolyReLU(3),
                'width': 64,
                'n_blocks': 2,
                'n_linear_drm': 2,
                'lr': 1e-3,
                'loss_weights': [1, 100, 100],
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
        case 'pinn': # optimized
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.SiLU(),
                'weight_init_kwargs': {'gain': 1.5},
                'width': 256,
                'depth': 10,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem4.source),
                    lf.Problem4.bc,
                ],
            })
        case 'drm': # optimized
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': torch.nn.SiLU(),
                'weight_init_kwargs': {'gain': 1.0},
                'width': 32,
                'n_blocks': 4,
                'n_linear_drm': 3,
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
        case 'pinn': # optimized
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.SiLU(),
                'width': 64,
                'depth': 8,
                'lr': 1e-2,
                'loss_fns': [
                    lf.PINN.laplacian(lf.Problem5.source),
                    lf.General.dirichlet_bc,
                    lf.Problem5.bc,
                ],
            })
        case 'drm': # optimized
            hyperparams.update({
                'model_constructor': 'drm',
                'act_fn': modules.PolyReLU(4),
                'width': 64,
                'n_blocks': 10,
                'lr': 1e-4,
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
        case 'pinn': # TODO
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
        case 'drm': # TODO
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


def NavierStokes2D(method: str = 'pinn') -> utils.Params:
    device = utils.get_device()
    v, rho, h, L, T, U_m = 1e-4, 1.0, 0.41, 2.2, 8, 1.8
    coord_space = utils.ParameterSpace([[0, L], [0, h], [0, T]], device)
    bc_select_top_bottom = torch.tensor([[0, 0], [1, 1], [0, 0]], device=device)
    # cylinder
    bc_select_inlet = torch.tensor([[1, 0], [0, 0], [0, 0]], device=device)
    bc_select_outlet = torch.tensor([[0, 1], [0, 0], [0, 0]], device=device)
    bc_select_ic = torch.tensor([[0, 0], [0, 0], [1, 0]], device=device)
    hyperparams: utils.Params = {
        'device': device,
        'n_in': 3,
        'n_out': 3,
        'weight_init_fn': torch.nn.init.xavier_uniform_,
        'bias_init_fn': torch.nn.init.zeros_,
        'weight_init_kwargs': {'gain': 0.5},
        'bias_init_kwargs': {},
        'optimizer': torch.optim.Adam,
        'lr': 1e-3,
        'n_epochs': 10000,
        'loss_fn_batch_sizes': [500, 500, 500, 500, 500, 500],
        'loss_weights': [1, 10, 10, 1, 1, 1],
        'loss_fn_data': [
            coord_space.rand(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_top_bottom),
            lf.NavierStokes2D.cylinder_bndry_data(100_000),
            coord_space.select_bndry_rand(100_000, bc_select_inlet),
            coord_space.select_bndry_rand(100_000, bc_select_outlet),
            coord_space.select_bndry_rand(100_000, bc_select_ic),
        ],
    }
    match method:
        case 'pinn':
            hyperparams.update({
                'model_constructor': 'rectangular_fnn',
                'act_fn': torch.nn.Tanh(),
                'width': 512,
                'depth': 8,
                'loss_fns': [
                    lf.PINN.navier_stokes_2d(viscosity=v, density=rho),
                    lf.NavierStokes2D.top_bottom_bc,
                    lf.NavierStokes2D.cylinder_bc,
                    lf.NavierStokes2D.inlet_bc(U_m, h),
                    lf.NavierStokes2D.outlet_bc,
                    lf.NavierStokes2D.initial_condition(device=device),
                ],
            })
        case _:
            raise ValueError(f'invalid method: {method}')
    return hyperparams