from project8.neural_network import models
from project8.neural_network.differential_operators import *
import torch


def dirichlet_bc_penalty(model: models.NN) -> torch.Tensor:
    return model.output.pow(2).mean()


def sinxy_source(input: torch.Tensor) -> torch.Tensor:
    return torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)


def wave_source(input: torch.Tensor) -> torch.Tensor:
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
    @staticmethod
    def nd_laplacian(model: models.NN) -> torch.Tensor:
        f = sinxy_source(model.input)
        laplace = laplacian(model)
        return (laplace + f).pow(2).mean()

    @staticmethod
    def wave_2d(model: models.NN) -> torch.Tensor:
        f = wave_source(model.input)
        residual = pd(model, 2, 2) - laplacian(model, time_dependent=True) - f
        return residual.pow(2).mean()


class DRM:
    @staticmethod
    def nd_laplacian(model: models.NN) -> torch.Tensor:
        grad = gradient(model)
        f = sinxy_source(model.input)
        return torch.mean(
            0.5 * torch.sum(grad.pow(2), 1).unsqueeze(1) - f * model.output)
