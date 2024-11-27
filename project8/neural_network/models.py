import torch
import torch.nn as nn
from torch.autograd import grad
from . import modules as mod
import warnings
from typing import Callable, Dict, Any, Optional, Self


class NN(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(
            self,
            weight_init: Callable[
                [torch.Tensor, Any], torch.Tensor
            ] | Callable[
                [torch.Tensor], torch.Tensor
            ],
            bias_init: Callable[[torch.Tensor, Any], torch.Tensor] | Callable[
                [torch.Tensor], torch.Tensor],
            weight_init_kwargs: Optional[Dict[str, Any]] = None,
            bias_init_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        if weight_init_kwargs is None:
            weight_init_kwargs = {}
        if bias_init_kwargs is None:
            bias_init_kwargs = {}
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_init(layer.weight, **weight_init_kwargs)
                bias_init(layer.bias, **bias_init_kwargs)
            if isinstance(layer, mod.DRM_block):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, nn.Linear):
                        weight_init(sub_layer.weight, **weight_init_kwargs)
                        bias_init(sub_layer.bias, **bias_init_kwargs)

    def __str__(self) -> str:
        out = ''
        for layer in self.layers:
            out = out + str(layer) + '\n'
        return out

    @classmethod
    def empty(cls) -> Self:
        return cls(nn.ModuleList())

    @classmethod
    def rectangular_fnn(cls,
                        n_in: int, n_out: int,
                        width: int, depth: int,
                        act_fn: nn.Module
                        ) -> Self:
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, width))
        layers.append(act_fn)
        for i in range(1, depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act_fn)
        layers.append(nn.Linear(width, n_out))
        return cls(layers)

    @classmethod
    def drm(cls,
            dim_in: int, dim_out: int,
            width: int, n_blocks: int,
            act_fn: nn.Module = mod.polyReLU(3),
            n_linear_drm: int = 2) -> Self:
        layers = nn.ModuleList()
        if dim_in != width:
            layers.append(nn.Linear(dim_in, width))
        if dim_in < width:
            warnings.warn(
                'padding input with zeros when n_in<width is not implemented'
            )
        for i in range(n_blocks):
            layers.append(mod.DRM_block(width, act_fn, n_linear_drm))
        layers.append(nn.Linear(width, dim_out))
        return cls(layers)


class diff_NN(NN):
    """
    Neural network with differential operators implemented,
    for convenience and efficiency writing pinn and drm loss functions.

    Differential operations are cached for efficiency,
     cache is reset at each forward pass.

    Dimension indexes (0 indexed) are used for partial derivatives.
    convention: last dimension is time, so xyzt, or xyz, or xyt etc.

    cache keys: out_dim_indexes + 'name' + in_dim_indexes
    (e.g. 0diff01 is grad(u_xy))
    """

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__(layers)
        self.input: torch.Tensor = torch.Tensor()
        self.output: torch.Tensor = torch.Tensor()
        self.__cache: Dict[
            str, torch.Tensor] = {}  # only access cache through methods

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.__cache = {}  # reset cached differences
        self.input = x
        self.output = super().forward(x)
        return self.output

    def output_append(self, cols: torch.Tensor) -> None:
        """Appends a new column to the output tensor.
        Needed for more complex pde's
        e.g. with derivatives wrt to u*v with u and v being outputs."""
        self.output = torch.cat((self.output, cols), 1)

    def gradient(self, out_dim_index: int = 0) -> torch.Tensor:
        """Returns gradient vector of the last output wrt last input"""
        key = str(out_dim_index) + 'diff'
        if key not in self.__cache:
            output = self.output[:, out_dim_index].unsqueeze(1)
            self.__cache[key] = grad(
                output, self.input, grad_outputs=torch.ones_like(output),
                create_graph=True
            )[0]
        return self.__cache[key]

    def diff(self, *in_dim_indexes: int,
             out_dim_index: int = 0) -> torch.Tensor:
        """Main method for getting partial derivatives.
        args are dimension indexes, e.g. diff(0, 1) is u_xy."""
        diff = self.gradient(out_dim_index)[:, in_dim_indexes[0]]
        key = str(out_dim_index) + 'diff'
        for i in range(1, len(in_dim_indexes)):
            key += str(in_dim_indexes[i - 1])
            if key not in self.__cache:
                self.__cache[key] = grad(
                    diff, self.input, grad_outputs=torch.ones_like(diff),
                    create_graph=True
                )[0]
            diff = self.__cache[key][:, in_dim_indexes[i]]
        return diff.unsqueeze(1)

    def divergence(self, vector: torch.Tensor) -> torch.Tensor:
        """Returns divergence of vector.
        Vector must be calculated from model input.
        Caching is not supported as function input can be anything"""
        if vector.shape == self.input.shape:
            div = torch.zeros_like(self.input[:, 0])
            for i in range(self.input.shape[1]):
                div += grad(
                    vector[:, i], self.input,
                    grad_outputs=torch.ones_like(vector[:, i]),
                    create_graph=True
                )[0][:, i]
            return div.unsqueeze(1)
        else:
            raise Exception('Output shape must be the same as input shape')

    def laplacian(self, out_dim_index: int = 0) -> torch.Tensor:
        """Returns laplacian of output,
        but the entire hessian is computed and cached.
        Not self.divergence(self.gradient(out_dim_index)),
        as that doesn't cache second diffs."""
        key = str(out_dim_index) + 'laplacian'
        if key not in self.__cache:
            laplacian_u = torch.zeros_like(self.output)
            for i in range(self.input.shape[1]):
                laplacian_u += self.diff(
                    i, i, out_dim_index=out_dim_index
                )
            self.__cache[key] = laplacian_u
        return self.__cache[key]
