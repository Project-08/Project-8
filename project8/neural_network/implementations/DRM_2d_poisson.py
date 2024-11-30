from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
import torch
import logging


def source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(4 * input[:, 0] * input[:, 1]).unsqueeze(1)


def drm_loss_2d_poisson_domain(model: models.diff_NN) -> torch.Tensor:
    # sum/mean of 0.5 * (u_x^2 + u_y^2) - f*u in the domain
    # eq 13 first term
    grad = model.gradient()
    f = source(model.input)
    return torch.mean(
        0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * model.output)


def drm_loss_2d_poisson_bndry(model: models.diff_NN) -> torch.Tensor:
    # sum/mean of u^2 on the boundary
    # eq 13 second term
    return model.output.pow(2).mean()


def train() -> None:
    device = utils.get_device()
    # init model
    act_fn = modules.Sin(torch.pi)
    model = models.diff_NN.drm(2, 1, 20, 4, act_fn=act_fn)
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
    coord_space = utils.ParameterSpace(
        [[-1, 2], [-1, 1]], device)
    tmr = utils.Timer()

    # plot source function
    f_loc = coord_space.fgrid(300)
    f = source(f_loc)
    x, y = coord_space.regrid(f_loc)
    f = coord_space.regrid(f)[0]
    utils.plot_2d(x, y, f, title='source function', fig_id=1)

    tmr.start()
    # train
    for epoch in range(n_epochs):
        model(coord_space.rand(1000).requires_grad_(True))
        domain_loss = drm_loss_2d_poisson_domain(model)
        model(coord_space.bndry_rand(500).requires_grad_(True))
        bndry_loss = drm_loss_2d_poisson_bndry(model)
        loss: torch.Tensor = domain_loss + 100 * bndry_loss
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
        if epoch % 100 == 0:
            logging.info(
                f'Epoch {epoch},'
                f' domain loss: {domain_loss.item()},'
                f' boundary loss: {bndry_loss.item()},'
                f' total: {loss.item()}')
    tmr.rr()

    # plot output
    grid = coord_space.fgrid(200)
    output = model(grid)
    x, y = coord_space.regrid(grid)
    f = coord_space.regrid(output)[0]
    utils.plot_2d(x, y, f, title='output_drm')


def main() -> None:
    train()
    pass


if __name__ == '__main__':
    main()
