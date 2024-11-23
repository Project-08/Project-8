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
    def simple_linear(cls, n_in: int, n_out: int, width: int, depth: int, act_fn: nn.Module):
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
    def DRM(cls, dim_in: int, width: int, n_blocks: int, act_fn: nn.Module = mod.polyReLU(3), n_linear_drm: int = 2):
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
    ''' Neural network with differential operators implemented,
    for convenience and efficiency writing pinn and drm loss functions.

    Differential operations are cached for efficiency, cache is reset at each forward pass.

    Dimension indexes are used for partial derivatives, e.g. first_diff(0) is du/dx, second_diff(0, 1) is d^2u/dxdy.
    convention: last dimension is time, so xyzt, or xyz, or xyt etc.'''
    def __init__(self, layers: nn.ModuleList):
        super().__init__(layers)
        self.input = None
        self.output = None
        self.__cache = {} # only access cache through methods

    def forward(self, x):
        self.__cache = {} # reset cached differences
        self.input = x
        self.output = super().forward(x)
        return self.output

    def gradient_vector(self):
        if 'gradient' not in self.__cache:
            self.__cache['gradient'] = torch.autograd.grad(self.output, self.input, grad_outputs=torch.ones_like(self.output), create_graph=True)[0]
        return self.__cache['gradient']

    def first_diff(self, dim_index):
        return self.gradient_vector()[:, dim_index].unsqueeze(1)

    def second_diff(self, dim_index_1, dim_index_2):
        cache_entry = 'second_diffs_'+str(dim_index_1)
        if cache_entry not in self.__cache:
            grad = self.gradient_vector()[:, dim_index_1]
            self.__cache[cache_entry] = torch.autograd.grad(
                grad, self.input, grad_outputs=torch.ones_like(grad), create_graph=True
            )[0]
        return self.__cache[cache_entry][:, dim_index_2].unsqueeze(1)

    def laplacian(self):
        # returns laplacian of output, but the entire hessian is computed and cached
        # possible efficiency improvement by not caching the entire hessian, but it has to be computed (I think)
        if 'laplacian' not in self.__cache:
            laplacian_u = torch.zeros_like(self.output)
            for i in range(self.input.shape[1]):
                laplacian_u += self.second_diff(i, i)
            self.__cache['laplacian'] = laplacian_u
            return laplacian_u # maybe more efficient to return here instead of looking up cache
        return self.__cache['laplacian']