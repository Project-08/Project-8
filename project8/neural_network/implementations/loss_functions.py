from project8.neural_network import models
from project8.neural_network.differential_operators import *
import torch
from typing import Callable


def dirichlet_bc_penalty(model: models.NN) -> torch.Tensor:
    return model.output.pow(2).mean()


def sinxy_source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)


def nmfde_a3_source(input: torch.Tensor) -> torch.Tensor:
    alpha = 40
    x = input[:, 0].unsqueeze(1)
    y = input[:, 1].unsqueeze(1)
    sum = torch.zeros_like(x)
    for i in range(1, 10):
        for j in range(1, 5):
            sum += torch.exp(-alpha * (x - i) ** 2 - alpha * (y - j) ** 2)
    return sum


def nmfde_a4_wave(input: torch.Tensor) -> torch.Tensor:
    Lx = 10
    Ly = 5
    alpha = 40
    omega = 4 * torch.pi
    f = torch.zeros_like(input[:, 0])
    a = [
        [0.25 * Lx, 0.25 * Ly],
        [0.25 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.25 * Ly]
    ]
    for i in a:
        f += torch.exp(
            -alpha * (input[:, 0] - i[0]) ** 2 - alpha * (
                    input[:, 1] - i[1]) ** 2)
    return (f * torch.sin(omega * input[:, 2])).unsqueeze(1)


class PINN:
    class nd_laplacian:
        def __init__(self, source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            f = self.source(model.input)
            laplace = laplacian(model)
            return (laplace + f).pow(2).mean()

    class wave_2d:
        def __init__(self, source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            t_dim_id = model.input.shape[1] - 1
            f = self.source(model.input)
            residual = pd(model, t_dim_id, t_dim_id) - laplacian(model, time_dependent=True) - f
            return residual.pow(2).mean()


class DRM:
    @staticmethod
    def nd_laplacian(model: models.NN) -> torch.Tensor:
        grad = gradient(model)
        f = sinxy_source(model.input)
        return torch.mean(
            0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * model.output)
