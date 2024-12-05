"""
File for defining custom modules.
Custom module names may not conflict with torch.nn module names!
"""

import torch
import torch.nn as nn
from typing import Any


class AbstractModule(nn.Module):
    """
    Abstract class for neural network modules

    Pass all arguments of __init__ as kwargs to super().__init__
    in child classes. This is to allow proper saving and loading.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__()
        self.kwargs: dict[str, Any] = kwargs

    def extra_repr(self) -> str:
        out = ''
        for key, value in self.kwargs.items():
            out = out + key + '=' + str(value) + ', '
        return out[:-2]


class PolyReLU(AbstractModule):
    """ReLU^n activation function (for DRM)"""

    def __init__(self, n: int) -> None:
        super().__init__(n=n)
        self.n: int = n

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.relu(x), self.n)


class Padding(AbstractModule):
    """Padding zeros along dim 1"""

    def __init__(self, to_width: int) -> None:
        super().__init__(to_width=to_width)
        self.width: int = to_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.pad(x, (0, self.width - x.shape[1]))


class DRM_block(AbstractModule):
    """DRM block as described in deep ritz paper, defaults are also in paper"""

    def __init__(self, width: int, act_fn: nn.Module = PolyReLU(3),
                 n_linear: int = 2) -> None:
        super().__init__(width=width, act_fn=act_fn, n_linear=n_linear)
        self.layers: nn.ModuleList = nn.ModuleList()
        for i in range(n_linear):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_temp: torch.Tensor = x
        for layer in self.layers:
            x_temp = layer(x_temp)
        return x + x_temp


class Sin(AbstractModule):
    """Sin activation function"""

    def __init__(self, freq: float = 1.0) -> None:
        super().__init__(freq=freq)
        self.freq: float = freq

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x * self.freq)


class FourierLike(AbstractModule):
    """
    Fourier series like activation function with learnable parameters
    for every neuron: x = a*sin(f*x + p), where a, f, p are learnable
    """

    def __init__(self, width: int) -> None:
        super().__init__(width=width)
        self.width: int = width
        self.amps: nn.Parameter = nn.Parameter(torch.randn(1, width))
        self.freqs: nn.Parameter = nn.Parameter(torch.randn(1, width))
        self.phases: nn.Parameter = nn.Parameter(torch.randn(1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.freqs * x + self.phases) * self.amps


class GaussianLike(AbstractModule):
    """
    Gaussian like activation function with learnable parameters
    for every neuron: x = a*exp(-b*(x - c)^2), where a, b, c are learnable
    """

    def __init__(self, width: int) -> None:
        super().__init__(width=width)
        self.width: int = width
        self.std: nn.Parameter = nn.Parameter(torch.randn(1, width))
        self.mean: nn.Parameter = nn.Parameter(torch.randn(1, width))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (
                self.std * torch.sqrt(2 * torch.tensor(torch.pi)))
