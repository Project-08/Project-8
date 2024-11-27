from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
import torch
import logging


def sourcefunc_A4(x, y, alpha=40):
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


def sourcefunc_wave(x, y, t, alpha=40, omega=4 * torch.pi):
    f = sourcefunc_A4(x, y, alpha)
    return f * torch.sin(omega * t)


def F(coords):
    return sourcefunc_wave(coords[:, 0], coords[:, 1], coords[:, 2]).unsqueeze(
        1)


def coeffK3(x, y):
    return 1 + 0.1 * (x + y + x * y)


def K(coords):
    return coeffK3(coords[:, 0], coords[:, 1]).unsqueeze(1)


def pinn_wave_pde(model: models.diff_NN):
    f = F(model.input)
    residual = model.diff(2, 2) - model.laplacian() - f
    return residual.abs().mean()


def pinn_wave_bc(model: models.diff_NN):
    return model.output.pow(2).mean()


def pinn_wave_ic(model: models.diff_NN):
    return model.output.pow(2).mean()


def train():
    device = utils.get_device()
    # init model
    act_fn = modules.FourierActivation(64)
    model = models.diff_NN.rectangular_fnn(3, 1, 64, 9, act_fn=act_fn)
    logging.info(model)
    model.to(device)
    model.double()
    model.initialize_weights(
        torch.nn.init.xavier_uniform_,
        torch.nn.init.zeros_,
        weight_init_kwargs={'gain': 0.5})

    # training params
    n_epochs = 5000
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    coord_space = utils.ParameterSpace([[0, 10], [0, 5], [0, 4]], device)
    tmr = utils.timer()
    domain_data_loader = utils.DataLoader(
        coord_space.rand(100000), 3000, device=device,
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

    # train
    tmr.start()
    for epoch in range(n_epochs):
        model(domain_data_loader())
        domain_loss = pinn_wave_pde(model)
        model(bc_data_loader())
        bndry_loss = pinn_wave_bc(model)
        model(ic_data_loader())
        ic_loss = pinn_wave_ic(model)
        loss = 100 * domain_loss + bndry_loss + ic_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            logging.info(f'Epoch {epoch},'
                         f' pde: {domain_loss.item()},'
                         f' bc: {bndry_loss.item()},'
                         f' ic: {ic_loss.item()},'
                         f' total: {loss.item()}'
                         )
    tmr.rr()

    # plot output
    grid = coord_space.fgrid(50)
    model.eval()
    output = model(grid)
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
