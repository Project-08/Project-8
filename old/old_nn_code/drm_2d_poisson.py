from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from old.old_nn_code import autograd_wrapper as aw
import torch
import logging
from typing import Any


def source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)


def drm_loss_2d_poisson_domain(diff: aw.Differentiator) -> torch.Tensor:
    grad = diff.gradient()
    f = source(diff.input())
    return torch.mean(
        0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * diff.output())


def drm_loss_2d_poisson_bndry(diff: aw.Differentiator) -> torch.Tensor:
    return diff.output().pow(2).mean()


def train() -> None:
    device = utils.get_device()
    # init model
    act_fn = modules.PolyReLU(3)
    model = models.NN.drm(2, 1, 20, 4, act_fn=act_fn)
    logging.info(model)
    model.to(device)
    model.double()
    model.initialize_weights(
        torch.nn.init.xavier_uniform_,
        torch.nn.init.zeros_,
        weight_init_kwargs={'gain': 0.5}
    )

    # training params
    n_epochs = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    diff = aw.Differentiator(model)
    coord_space = utils.ParameterSpace(
        [[-1, 2], [-1, 1]], device)
    pde_data = utils.DataLoader(coord_space.rand(100000), 1000, True)
    bndry_data = utils.DataLoader(coord_space.bndry_rand(100000), 500, True)
    problem: list[list[Any]] = [
        [drm_loss_2d_poisson_domain, pde_data, 1],
        [drm_loss_2d_poisson_bndry, bndry_data, 10]
    ]

    # plot source function
    f_loc = coord_space.fgrid(300)
    f = source(f_loc)
    x, y = coord_space.regrid(f_loc)
    f = coord_space.regrid(f)[0]
    utils.plot_2d(x, y, f, title='source_drm_2d_poisson', fig_id=1)
    # train
    for epoch in range(n_epochs):
        loss: torch.Tensor = torch.tensor(0.0, device=device)
        for i in problem:
            diff.set_input(i[1]())
            loss += i[2] * i[0](diff)
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, loss: {loss.item()}')

    # plot output
    grid = coord_space.fgrid(200)
    output = model(grid)
    x, y = coord_space.regrid(grid)
    f = coord_space.regrid(output)[0]
    utils.plot_2d(x, y, f, title='output_drm_2d_poisson', fig_id=2)


def main() -> None:
    tmr = utils.Timer()
    tmr.start()
    train()
    tmr.rr()
    pass


if __name__ == '__main__':
    main()
