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
        if weight_init_kwargs is None: weight_init_kwargs = {}
        if bias_init_kwargs is None: bias_init_kwargs = {}
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

    # rectangular FNN constructor (for PINN)
    @classmethod
    def rectangular_fnn(cls, n_in: int, n_out: int, width: int, depth: int, act_fn: nn.Module):
        layers = nn.ModuleList()
        layers.append(nn.Linear(n_in, width))
        layers.append(act_fn)
        for i in range(1, depth-1):
            layers.append(nn.Linear(width, width))
            layers.append(act_fn)
        layers.append(nn.Linear(width, n_out))
        return cls(layers)

    # DRM NN constructor
    @classmethod
    def drm(cls, dim_in: int, width: int, n_blocks: int, act_fn: nn.Module = mod.polyReLU(3), n_linear_drm: int = 2):
        layers = nn.ModuleList()
        if dim_in != width:
            layers.append(nn.Linear(dim_in, width))
        if dim_in < width:
            warnings.warn('padding input with zeros when n_in<width is not implemented (described in paper)')
        for i in range(n_blocks):
            layers.append(mod.DRM_block(width, act_fn, n_linear_drm))
        layers.append(nn.Linear(width, 1))
        return cls(layers)


class diff_NN(NN):
    """ Neural network with differential operators implemented,
    for convenience and efficiency writing pinn and drm loss functions.

    Differential operations are cached for efficiency, cache is reset at each forward pass.

    Dimension indexes (0 indexed) are used for partial derivatives.
    convention: last dimension is time, so xyzt, or xyz, or xyt etc."""
    def __init__(self, layers: nn.ModuleList):
        super().__init__(layers)
        self.input = None
        self.output = None
        # self.use_cache = True
        self.__cache = {} # only access cache through methods

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.__cache = {} # reset cached differences
        self.input = x
        self.output = super().forward(x)
        return self.output

    def gradient(self) -> torch.Tensor:
        """Returns gradient vector of the last output wrt last input"""
        if 'gradient' not in self.__cache:
            self.__cache['gradient'] = torch.autograd.grad(self.output, self.input, grad_outputs=torch.ones_like(self.output), create_graph=True)[0]
        return self.__cache['gradient']

    def diff(self, *dim_indexes: int) -> torch.Tensor:
        """Main method for getting partial derivatives.
        args are dimension indexes, e.g. diff(0, 1) is u_xy."""
        diff = self.gradient()[:, dim_indexes[0]]
        key = 'diff'
        for i in range(1, len(dim_indexes)):
            key += str(dim_indexes[i-1])
            if key not in self.__cache:
                self.__cache[key] = torch.autograd.grad(diff, self.input, grad_outputs=torch.ones_like(diff), create_graph=True)[0]
            diff = self.__cache[key][:, dim_indexes[i]]
        return diff.unsqueeze(1)

    def laplacian(self) -> torch.Tensor:
        """Returns laplacian of output, but the entire hessian is computed and cached"""
        if 'laplacian' not in self.__cache:
            laplacian_u = torch.zeros_like(self.output)
            for i in range(self.input.shape[1]):
                laplacian_u += self.diff(i, i)
            self.__cache['laplacian'] = laplacian_u
        return self.__cache['laplacian']

    # maybe more (differential) operators