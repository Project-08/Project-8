from project8.neural_network import models
from project8.neural_network import differential_operators as diffop
import torch
from typing import Callable


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


def zero_source(input: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(input[:, 0]).unsqueeze(1)

def problem1_source(input: torch.Tensor) -> torch.Tensor:
    # not negative as in the paper because the loss fn is for -laplacian(u) = f
    return (torch.pi ** 2) * input[:, 0] * input[:, 1] * torch.sin(torch.pi * input[:, 2])


def problem_1_exact(input: torch.Tensor) -> torch.Tensor:
    return input[:, 0] * input[:, 1] * torch.sin(torch.pi * input[:, 2])


class PINN:
    class nd_laplacian:
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            f = self.source(model.input)
            laplace = diffop.laplacian(model, time_dependent=False)
            return (laplace + f).pow(2).mean()

    class nd_wave:
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            t_dim_id = model.input.shape[1] - 1
            f = self.source(model.input)
            residual = diffop.partial_derivative(model, t_dim_id,
                                                 t_dim_id) - diffop.laplacian(
                model,
                time_dependent=True) - f
            return residual.pow(2).mean()


class DRM:
    class nd_laplacian:
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            grad = diffop.gradient(model)
            f = self.source(model.input)
            return torch.mean(
                0.5 * torch.sum(grad.pow(2), 1).unsqueeze(
                    1) - f * model.output)


class General:
    class constant_bc_penalty:
        def __init__(self, constant: float = 0) -> None:
            self.target: float = constant

        def __call__(self, model: models.NN) -> torch.Tensor:
            return (model.output - self.target).pow(2).mean()

    class dirichlet_bc_penalty:
        def __call__(self, model: models.NN) -> torch.Tensor:
            return model.output.pow(2).mean()

    class fit_to_fn:
        def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.fn = fn

        def __call__(self, model: models.NN) -> torch.Tensor:
            return (model.output - self.fn(model.input)).pow(2).mean()

    class problem1_bc_x:
        def __call__(self, model: models.NN) -> torch.Tensor:
            residual = model.output - model.input[:, 1] * torch.sin(torch.pi * model.input[:, 2])
            return residual.pow(2).mean()

    class problem1_bc_y:
        def __call__(self, model: models.NN) -> torch.Tensor:
            residual = model.output - model.input[:, 0] * torch.sin(torch.pi * model.input[:, 2])
            return residual.pow(2).mean()
