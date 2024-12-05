"""
File for defining models.
model names may not conflict with torch.nn module names!
"""

import torch
import torch.nn as nn
from project8.neural_network import modules as mod
from project8.neural_network.utils import Params
from typing import Callable, Dict, Any, Optional, Self, TypedDict


class NN(nn.Module):
    """
    Sequential model class.
    saves input, output, and cache to be able to compute derivatives.
    """

    def __init__(self, layers: nn.ModuleList,
                 device: str | torch.device = 'cpu') -> None:
        super().__init__()
        self.layers: nn.ModuleList = layers
        self.double()
        self.to(device)
        self.input: torch.Tensor = torch.Tensor()
        self.output: torch.Tensor = torch.Tensor()
        self.cache: dict[str, torch.Tensor] = {}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.cache = {}
        self.input = x
        self.output = self.layers[0](x)
        for layer in self.layers[1:]:
            self.output = layer(self.output)
        return self.output

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
        Should return a string that can be evaluated to initialize a model.
        """
        out = self.__class__.__name__ + '(ModuleList([\n'
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
            out += '    ' + layer_str + ',\n'
        return out[:-2] + '\n]))'

    @classmethod
    def empty(cls, device: str | torch.device = 'cpu') -> Self:
        return cls(nn.ModuleList(), device)

    @classmethod
    def rectangular_fnn(cls,
                        n_in: int, n_out: int,
                        width: int, depth: int,
                        act_fn: nn.Module,
                        device: str | torch.device = 'cpu'
                        ) -> Self:
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, width))
        layers.append(act_fn)
        for i in range(1, depth - 1):
            layers.append(nn.Linear(width, width))
            layers.append(act_fn)
        layers.append(nn.Linear(width, n_out))
        return cls(layers, device)

    @classmethod
    def drm(cls,
            n_in: int, n_out: int,
            width: int, n_blocks: int,
            act_fn: nn.Module = mod.PolyReLU(3),
            n_linear_drm: int = 2,
            device: str | torch.device = 'cpu'
            ) -> Self:
        layers = nn.ModuleList()
        if n_in > width:
            layers.append(nn.Linear(n_in, width))
        elif n_in < width:
            layers.append(mod.Padding(width))
        for i in range(n_blocks):
            layers.append(mod.DRM_block(width, act_fn, n_linear_drm))
        layers.append(nn.Linear(width, n_out))
        return cls(layers, device)

    @classmethod
    def from_param_dict(cls, params: Params) -> Self:
        match params['model_constructor']:
            case 'empty':
                return cls.empty(
                    params['device']
                )
            case 'rectangular_fnn':
                model = cls.rectangular_fnn(
                    params['n_in'], params['n_out'],
                    params['width'], params['depth'],
                    params['act_fn'], params['device']
                )
            case 'drm':
                model = cls.drm(
                    params['n_in'], params['n_out'],
                    params['width'], params['n_blocks'],
                    params['act_fn'], params['n_linear_drm'],
                    params['device']
                )
            case _:
                raise ValueError('Invalid model_constructor_name')
        if 'weight_init_fn' in params and 'bias_init_fn' in params:
            model.initialize_weights(
                params['weight_init_fn'],
                params['bias_init_fn'],
                weight_init_kwargs=params['weight_init_kwargs'],
                bias_init_kwargs=params['bias_init_kwargs']
            )
        return model


def main() -> None:
    model = NN.drm(2, 1, 64, 9)
    print(model.definition())
    pass


if __name__ == '__main__':
    main()
