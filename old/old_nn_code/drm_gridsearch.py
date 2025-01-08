from project8.neural_network import utils
from project8.neural_network import modules
from project8.neural_network import models
from old.old_nn_code import autograd_wrapper as aw
import torch
import logging
from typing import Any


def source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(4 * input[:, 0] * input[:, 1]).unsqueeze(1)


def drm_loss_2d_poisson_domain(diff: aw.Differentiator) -> torch.Tensor:
    grad = diff.gradient()
    f = source(diff.input())
    return torch.mean(
        0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * diff.output())


def drm_loss_2d_poisson_bndry(diff: aw.Differentiator) -> torch.Tensor:
    return diff.output().pow(2).mean()


def train(params: dict[str, Any]) -> tuple[models.NN, float]:
    device = params['device']
    # init model
    act_fn = params['act_fn']
    model = models.NN.drm(
        params['n_inputs'],
        params['n_outputs'],
        params['width'],
        params['n_blocks'],
        n_linear_drm=params['n_linear_per_block'],
        act_fn=act_fn)
    logging.info(model)
    model.to(device)
    model.double()
    model.initialize_weights(
        params['weight_init_fn'],
        params['bias_init_fn'],
        weight_init_kwargs=params['weight_init_kwargs']
    )

    # training params
    optimizer = params['optimizer'](model.parameters(), lr=params['lr'])
    diff = aw.Differentiator(model)

    # train
    for epoch in range(params['n_epochs']):
        loss: torch.Tensor = torch.tensor(0.0, device=device)
        for dl, loss_fn, weight in zip(
                params['data_loaders'],
                params['loss_fns'],
                params['loss_weights']):
            diff.forward(dl())
            loss += weight * loss_fn(diff)
        optimizer.zero_grad()
        loss.backward()  # type: ignore
        optimizer.step()
    return model, loss.item()


def main() -> None:
    device = utils.get_device()
    space = utils.ParameterSpace([[-1, 2], [-1, 1]], device)
    dl1 = utils.DataLoader(space.rand(1000), 1000, True)
    dl2 = utils.DataLoader(space.bndry_rand(500), 500, True)
    params: dict[str, Any] = {
        'loss_fns': [drm_loss_2d_poisson_domain, drm_loss_2d_poisson_bndry],
        'loss_weights': [1, 100],
        'data_loaders': [dl1, dl2],
        'device': device,
        'n_inputs': 2,
        'n_outputs': 1,
        'width': 20,  # hyperparam
        'n_blocks': 4,  # hyperparam
        'n_linear_per_block': 2,  # hyperparam
        'act_fn': modules.Sin(torch.pi),  # hyperparam
        'weight_init_fn': torch.nn.init.xavier_uniform_,  # hyperparam
        'bias_init_fn': torch.nn.init.zeros_,  # hyperparam
        'weight_init_kwargs': {'gain': 0.5},  # hyperparam
        'n_epochs': 5000,  # hyperparam
        'optimizer': torch.optim.Adam,  # hyperparam
        'lr': 1e-3,  # hyperparam
    }
    model, loss = train(params)
    pass


if __name__ == '__main__':
    main()
