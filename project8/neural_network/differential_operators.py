import torch
from torch.autograd import grad
from project8.neural_network.models import NN
from typing import Dict, Callable

class AbstractIO:
    """
    Only used for tests.
    """
    input: torch.Tensor
    output: torch.Tensor
    cache: Dict[str, torch.Tensor]
    def __init__(self) -> None:
        self.cache = {}
        self.input = torch.Tensor()
        self.output = torch.Tensor()

"""
set type hint of 'io' for all functions
should be object with input and output attributes that are Tensors
and a cache attribute that is a dictionary
this object represents a trial solution
"""
arg_type = NN | AbstractIO


def gradient(
        io: arg_type,
        out_dim_index: int = 0
) -> torch.Tensor:
    """Returns gradient vector of the output wrt input"""
    key = str(out_dim_index) + 'diff'
    if key not in io.cache:
        output = io.output[:, out_dim_index].unsqueeze(1)
        io.cache[key] = grad(
            output,
            io.input,
            grad_outputs=torch.ones_like(output),
            create_graph=True)[0]
    return io.cache[key]


def partial_derivative(
        io: arg_type,
        *in_dim_indexes: int,
        out_dim_index: int = 0
) -> torch.Tensor:
    """Main method for getting partial derivatives.
    args are dimension indexes, e.g. diff(0, 1) is u_xy."""
    diff = gradient(io, out_dim_index)[:, in_dim_indexes[0]]
    key = str(out_dim_index) + 'diff'
    for i in range(1, len(in_dim_indexes)):
        key += str(in_dim_indexes[i - 1])
        if key not in io.cache:
            io.cache[key] = grad(
                diff, io.input,
                grad_outputs=torch.ones_like(diff),
                create_graph=True
            )[0]
        diff = io.cache[key][:, in_dim_indexes[i]]
    return diff.unsqueeze(1)


def pd(
        io: arg_type,
        *in_dim_indexes: int,
        out_dim_index: int = 0
) -> torch.Tensor:
    return partial_derivative(io, *in_dim_indexes, out_dim_index=out_dim_index)


def divergence(
        io: arg_type,
        vector: torch.Tensor
) -> torch.Tensor:
    """
    Returns divergence of vector.
    gradients will be calculated wrt input of io.
    no caching is supported as function input can be anything
    """

    if vector.shape == io.input.shape:
        div = torch.zeros_like(vector[:, 0])
        for i in range(vector.shape[1]):
            div += grad(
                vector[:, i], io.input,
                grad_outputs=torch.ones_like(vector[:, i]),
                create_graph=True
            )[0][:, i]
        return div.unsqueeze(1)
    else:
        raise Exception('Output shape must be the same as input shape')


def laplacian(
        io: arg_type,
        out_dim_index: int = 0,
        time_dependent: bool = False
) -> torch.Tensor:
    """
    Returns laplacian of output,
    but the entire hessian is computed and cached.
    Not divergence(gradient(out_dim_index)),
    as that doesn't cache second diffs.
    """
    laplacian_u = torch.zeros_like(io.output[:, 0].unsqueeze(1))
    dims = io.input.shape[1] if not time_dependent else io.input.shape[1] - 1
    for i in range(dims):
        laplacian_u += partial_derivative(
            io,
            i, i,
            out_dim_index=out_dim_index
        )
    return laplacian_u
