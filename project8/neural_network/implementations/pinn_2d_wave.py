from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from project8.neural_network import autograd_wrapper as aw
import torch
import logging
from typing import Any


def sourcefunc_A4(x: torch.Tensor, y: torch.Tensor,
                  alpha: float = 40) -> torch.Tensor:
    Lx = 10
    Ly = 5
    f = torch.zeros_like(x)
    a = [
        [0.25 * Lx, 0.25 * Ly],
        [0.25 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.25 * Ly]
    ]
    for i in a:
        f += torch.exp(-alpha * (x - i[0]) ** 2 - alpha * (y - i[1]) ** 2)
    return f


def sourcefunc_wave(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor,
                    alpha: float = 40,
                    omega: float = 4 * torch.pi) -> torch.Tensor:
    f = sourcefunc_A4(x, y, alpha)
    return f * torch.sin(omega * t)


def F(coords: torch.Tensor) -> torch.Tensor:
    return sourcefunc_wave(coords[:, 0], coords[:, 1], coords[:, 2]).unsqueeze(
        1)


def coeffK3(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 1 + 0.1 * (x + y + x * y)


def K(coords: torch.Tensor) -> torch.Tensor:
    return coeffK3(coords[:, 0], coords[:, 1]).unsqueeze(1)


def pinn_wave_pde(diff: aw.Differentiator) -> torch.Tensor:
    f = F(diff.input())
    residual: torch.Tensor = 2 * diff(2, 2) - diff.laplacian() - f
    # .laplacian also includes 2nd diff wrt time
    return residual.pow(2).mean()


def pinn_wave_bc(diff: aw.Differentiator) -> torch.Tensor:
    return diff.output().pow(2).mean()


def pinn_wave_ic(diff: aw.Differentiator) -> torch.Tensor:
    return diff.output().pow(2).mean()


def fit_to_source(diff: aw.Differentiator) -> torch.Tensor:
    return (diff.output() - F(diff.input())).pow(2).mean()


def train() -> None:
    device = utils.get_device()
    # init model
    act_fn = torch.nn.Tanh()
    model = models.NN.rectangular_fnn(3, 1, 64, 9, act_fn=act_fn)
    logging.info(model)
    model.to(device)
    model.double()
    model.initialize_weights(
        torch.nn.init.xavier_uniform_,
        torch.nn.init.zeros_,
        weight_init_kwargs={'gain': 0.7})

    # training params
    n_epochs = 10000
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 20, gamma=0.99)
    diff = aw.Differentiator(function_to_attach=model)
    coord_space = utils.ParameterSpace([[0, 5], [0, 2.5], [0, 1]], device)
    domain_data_loader = utils.DataLoader(
        coord_space.rand(100_000_000), 10000, device=device,
        output_requires_grad=True)
    bc_select = torch.tensor([[1, 1], [1, 1], [0, 0]], device=device,
                             dtype=torch.float64)
    bc_data_loader = utils.DataLoader(
        coord_space.select_bndry_rand(100000, bc_select), 800, device=device,
        output_requires_grad=True)
    ic_select = torch.tensor([[0, 0], [0, 0], [1, 0]], device=device,
                             dtype=torch.float64)
    ic_data_loader = utils.DataLoader(
        coord_space.select_bndry_rand(100000, ic_select), 300, device=device,
        output_requires_grad=True)
    grid_dl = utils.DataLoader(coord_space.fgrid(50), 50, device=device,
                               output_requires_grad=True)
    problem: list[list[Any]] = [
        [pinn_wave_pde, domain_data_loader, 1],
        [pinn_wave_bc, bc_data_loader, 1],
        [pinn_wave_ic, ic_data_loader, 1]
    ]
    problem_2: list[list[Any]] = [
        [fit_to_source, domain_data_loader, 1]
    ]

    # train
    for epoch in range(n_epochs):
        loss: torch.Tensor = torch.tensor(0.0, device=device)
        for i in problem_2:
            diff.set_input(i[1]())
            loss += i[2] * i[0](diff)
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
        lr_scheduler.step()
        if epoch % 100 == 0:
            print(
                f'Epoch {epoch}, loss: {loss.item()}, '
                f'lr: {lr_scheduler.get_last_lr()}')

    # plot output
    grid = coord_space.fgrid(50)
    model.eval()
    output = model(grid)
    source_fn = F(grid)
    x, y, t = coord_space.regrid(grid)
    f = coord_space.regrid(output)[0]

    t_id = 0
    utils.plot_2d(x[t_id, :, :], y[t_id, :, :], f[t_id, :, :],
                  title='output_wave1', fig_id=2)
    t_id = 20
    utils.plot_2d(x[t_id, :, :], y[t_id, :, :], f[t_id, :, :],
                  title='output_wave2', fig_id=3)
    t_id = 49
    utils.plot_2d(x[t_id, :, :], y[t_id, :, :], f[t_id, :, :],
                  title='output_wave3', fig_id=4)


def main() -> None:
    tmr = utils.Timer()
    tmr.start()
    train()
    tmr.rr()
    pass


if __name__ == '__main__':
    main()
