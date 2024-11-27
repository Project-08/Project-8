from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
import torch
import logging


def source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)


def pinn_domain_loss(model: models.diff_NN) -> torch.Tensor:
    f = source(model.input)
    laplace = model.laplacian()
    return (laplace + f).pow(2).mean()


def pinn_bndry_loss(model: models.diff_NN) -> torch.Tensor:
    return model.output.pow(2).mean()


def train() -> None:
    device = utils.get_device()
    # init model
    act_fn = modules.FourierActivation(64)
    model = models.diff_NN.rectangular_fnn(2, 1, 64, 9, act_fn=act_fn)
    logging.info(model)
    model.to(device)
    model.double()
    model.initialize_weights(
        torch.nn.init.xavier_uniform_, torch.nn.init.zeros_,
        weight_init_kwargs={'gain': 0.5}
    )

    # training params
    n_epochs = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    coord_space = utils.ParameterSpace([[-1, 2], [-1, 1]],
                                       device)  # domain defined here
    tmr = utils.Timer()

    # plot source function
    f_loc = coord_space.fgrid(300)
    f = source(f_loc)
    x, y = coord_space.regrid(f_loc)
    f = coord_space.regrid(f)[0]
    utils.plot_2d(x, y, f, title='source_pinn_2d_poisson', fig_id=1)

    # train
    tmr.start()
    for epoch in range(n_epochs):
        model(coord_space.rand(1000).requires_grad_(
            True))  # forward pass on domain
        domain_loss = pinn_domain_loss(model)  # loss before other forward pass
        model(coord_space.bndry_rand(500).requires_grad_(
            True))  # forward pass on boundary
        bndry_loss = pinn_bndry_loss(model)
        loss: torch.Tensor = domain_loss + 1 * bndry_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logging.info(
                f'Epoch {epoch},'
                f' domain loss: {domain_loss.item()}, '
                f'boundary loss: {bndry_loss.item()}, '
                f'total: {loss.item()}')
    tmr.rr()

    # plot output
    grid = coord_space.fgrid(200)
    output = model(grid)
    x, y = coord_space.regrid(grid)
    f = coord_space.regrid(output)[0]
    utils.plot_2d(x, y, f, title='output_pinn_2d_poisson', fig_id=2)
