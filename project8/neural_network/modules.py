import torch
import torch.nn as nn


class polyReLU(nn.Module):
    """ReLU^n activation function (for DRM)"""

    def __init__(self, n: int):
        super().__init__()
        self.n = n

    def forward(self, x):
        return torch.pow(torch.relu(x), self.n)

    def __str__(self):
        return 'polyReLU(n=' + str(self.n) + ')'


class DRM_block(nn.Module):
    """DRM block as described in deep ritz paper, defaults are also in paper"""

    def __init__(self, width: int, act_fn: nn.Module = polyReLU(3),
                 n_linear: int = 2):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(n_linear):
            self.layers.append(nn.Linear(width, width))
            self.layers.append(act_fn)

    def forward(self, x):
        x_temp = x
        for layer in self.layers:
            x_temp = layer(x_temp)
        return x + x_temp

    def __str__(self):
        out = 'DRM_block:\n'
        for layer in self.layers:
            out = out + '   ' + str(layer) + '\n'
        return out + '   ' + '+ block input (skip/residual connection)'


class Sin(nn.Module):
    """Sin activation function"""

    def __init__(self, multiplier=1):
        super().__init__()
        self.multiplier = multiplier

    def forward(self, x):
        return torch.sin(x * self.multiplier)

    def __str__(self):
        return f'Sin(multiplier={self.multiplier})'


class FourierActivation(nn.Module):
    """Fourier series like activation function with learnable parameters"""

    def __init__(self, width: int):
        super().__init__()
        self.width = width
        self.amps = nn.Parameter(torch.randn(1, width))
        self.freqs = nn.Parameter(torch.randn(1, width))
        self.phases = nn.Parameter(torch.randn(1, width))

    def forward(self, x):
        return torch.sin(self.freqs * x + self.phases) * self.amps

    def __str__(self):
        return self.__class__.__name__ + '(width=' + str(self.width) + ')'
