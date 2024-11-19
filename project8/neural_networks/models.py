import torch
import torch.nn as nn
from . import modules as mod
import warnings


class NN(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def initialize_weights(self, weight_init, bias_init, weight_init_kwargs=None, bias_init_kwargs=None):
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

    def __str__(self):
        out = ''
        for layer in self.layers:
            out = out + str(layer) +'\n'
        return out

    # rectangular linear FNN constructor (for PINN)
    @staticmethod
    def simple_linear(n_in: int, n_out: int, width: int, depth: int, act_fn: nn.Module):
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, width))
        layers.append(act_fn)
        for i in range(1, depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(act_fn)
        layers.append(nn.Linear(width, n_out))
        return NN(layers)

    # DRM NN constructor
    @staticmethod
    def DRM(dim_in: int, width: int, n_blocks: int, act_fn: nn.Module = mod.polyReLU(3), n_linear_drm: int = 2):
        layers = nn.ModuleList()
        if dim_in != width:
            layers.append(nn.Linear(dim_in, width))
        if dim_in < width:
            warnings.warn('padding input with zeros when n_in<width is not implemented')
        for i in range(n_blocks):
            layers.append(mod.DRM_block(width, act_fn, n_linear_drm))
        layers.append(nn.Linear(width, 1))
        return NN(layers)

