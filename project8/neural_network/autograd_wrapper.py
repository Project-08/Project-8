import torch
from torch.autograd import grad
from typing import Dict, Callable


class Differentiator:
    """
    Differentiator class for neural networks.

    Differential operations are cached for efficiency,
     cache is reset at each forward pass.

    Dimension indexes (0 indexed) are used for partial derivatives.
    convention: last dimension is time, so xyzt, or xyz, or xyt etc.

    cache keys: out_dim_indexes + 'name' + in_dim_indexes
    (e.g. 0diff01 is grad(u_xy))
    """

    def __init__(self, function_to_attach: Callable[
        [torch.Tensor], torch.Tensor]
    ) -> None:
        self.function = function_to_attach
        self.__input: torch.Tensor = torch.Tensor()
        self.__output: torch.Tensor = torch.Tensor()
        self.__cache: Dict[str, torch.Tensor] = {}  # private cache

    def set_input(self, x: torch.Tensor) -> None:
        """Sets input tensor and executes function"""
        self.__cache = {}  # reset cached differences
        self.__input = x
        self.__output = self.function(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sets input tensor and executes function"""
        self.set_input(x)
        return self.__output

    def input(self) -> torch.Tensor:
        """Returns input tensor"""
        return self.__input

    def output(self) -> torch.Tensor:
        """Returns output tensor"""
        return self.__output

    def output_append(self, cols: torch.Tensor) -> None:
        """Appends a new column to the output tensor.
        Needed for more complex pde's
        e.g. with derivatives wrt to u*v with u and v being outputs."""
        self.__output = torch.cat((self.__output, cols), 1)

    def gradient(self, out_dim_index: int = 0) -> torch.Tensor:
        """Returns gradient vector of the output wrt input"""
        key = str(out_dim_index) + 'diff'
        if key not in self.__cache:
            output = self.__output[:, out_dim_index].unsqueeze(1)
            self.__cache[key] = grad(
                output, self.__input, grad_outputs=torch.ones_like(output),
                create_graph=True
            )[0]
        return self.__cache[key]

    def __call__(self, *in_dim_indexes: int,
                 out_dim_index: int = 0) -> torch.Tensor:
        """Main method for getting partial derivatives.
        args are dimension indexes, e.g. diff(0, 1) is u_xy."""
        diff = self.gradient(out_dim_index)[:, in_dim_indexes[0]]
        key = str(out_dim_index) + 'diff'
        for i in range(1, len(in_dim_indexes)):
            key += str(in_dim_indexes[i - 1])
            if key not in self.__cache:
                self.__cache[key] = grad(
                    diff, self.__input, grad_outputs=torch.ones_like(diff),
                    create_graph=True
                )[0]
            diff = self.__cache[key][:, in_dim_indexes[i]]
        return diff.unsqueeze(1)

    def divergence(self, vector: torch.Tensor) -> torch.Tensor:
        """
        Returns divergence of vector.
        Vector must be calculated from model input.
        Caching is not supported as function input can be anything
        """

        if vector.shape == self.__input.shape:
            div = torch.zeros_like(self.__input[:, 0])
            for i in range(self.__input.shape[1]):
                div += grad(
                    vector[:, i], self.__input,
                    grad_outputs=torch.ones_like(vector[:, i]),
                    create_graph=True
                )[0][:, i]
            return div.unsqueeze(1)
        else:
            raise Exception('Output shape must be the same as input shape')

    def laplacian(self, out_dim_index: int = 0) -> torch.Tensor:
        """
        Returns laplacian of output,
        but the entire hessian is computed and cached.
        Not self.divergence(self.gradient(out_dim_index)),
        as that doesn't cache second diffs.
        """

        key = str(out_dim_index) + 'laplacian'
        if key not in self.__cache:
            laplacian_u = torch.zeros_like(self.__output)
            for i in range(self.__input.shape[1]):
                laplacian_u += self.__call__(
                    i, i, out_dim_index=out_dim_index
                )
            self.__cache[key] = laplacian_u
        return self.__cache[key]
