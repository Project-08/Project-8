"""
File for defining models.
model names may not conflict with torch.nn module names!
"""

import torch
import torch.nn as nn
from project8.neural_network import modules as mod
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
            bias_init: Callable[
                           [torch.Tensor, Any], torch.Tensor
                       ] | Callable[
                           [torch.Tensor], torch.Tensor
                       ],
            weight_init_kwargs: Optional[Dict[str, Any]] = None,
            bias_init_kwargs: Optional[Dict[str, Any]] = None
    ) -> None:
        if weight_init_kwargs is None:
            weight_init_kwargs = {}
        if bias_init_kwargs is None:
            bias_init_kwargs = {}
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                weight_init(layer.weight, **weight_init_kwargs)  # type: ignore
                bias_init(layer.bias, **bias_init_kwargs)  # type: ignore
            if isinstance(layer, mod.DRM_block):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, nn.Linear):
                        weight_init(sub_layer.weight,
                                    **weight_init_kwargs)  # type: ignore
                        bias_init(sub_layer.bias,
                                  **bias_init_kwargs)  # type: ignore

    def definition(self) -> str:
        """
        Returns a string with enough information to reconstruct the model.
        It should be multiple lines of executable code that will return the
        various layers of the model.
        """
        out = ''
        for layer in self.layers:
            layer_str: str = ''
            lines: list[str] = str(layer).split('\n')
            for line in lines:
                if ':' not in line:
                    layer_str += line.strip()
            while layer_str.count('(') > layer_str.count(')'):
                layer_str += ')'
            while layer_str.count(')') > layer_str.count('('):
                layer_str = '(' + layer_str
            while layer_str.startswith('(') and layer_str.endswith(')'):
                layer_str = layer_str[1:-1]
            out += layer_str
            out += '\n'
        return out[:-1]

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
