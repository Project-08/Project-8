"""This file contains some convenience functions for the project"""

import torch
import matplotlib.pyplot as plt
from warnings import warn
import time
import os
import logging

plot_folder = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ), 'plots')
logging.info(f'plot_folder: {plot_folder}')


def get_device(override: str = None) -> str:
    if override is not None:
        return override
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warn('CUDA/ROCm not available, Using CPU.')
    return device


def if_tensors_to_numpy(*args) -> list:
    out = []
    for arg in args:
        if type(arg) == torch.Tensor:
            out.append(arg.detach().to('cpu').numpy())
        else:
            out.append(arg)
    return out


def plot_2d(x, y, f, title, fig_id=0, filename=None) -> None:
    x, y, f = if_tensors_to_numpy(x, y, f)
    if filename is None:
        filename = title + '.png'
    plt.figure(fig_id)
    plt.clf()
    domain = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(f, extent=domain, origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if domain[1] - domain[0] < 1.1 * (domain[3] - domain[2]):
        plt.colorbar(orientation='vertical')
    else:
        plt.colorbar(orientation='horizontal')
    plt.savefig(plot_folder + '/' + filename)


def plot_plane(x, y, f, title, fig_id=0, filename=None) -> None:
    x, y, f = if_tensors_to_numpy(x, y, f)
    if filename is None:
        filename = title + '.png'
    fig = plt.figure(fig_id)
    plt.clf()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, f, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    ax.set_title(title)
    plt.savefig(plot_folder + '/' + filename)


class ParameterSpace:
    def __init__(self, domain: iter, device='cpu'):
        """
        Initialize the float_parameter_space class.

        Parameters:
        domain (iter): A 2D tensor of floats, for constant parameters, use [a, a].
        device (str): The device to use ('cpu' or 'cuda').

        Note:
        Most methods output a tensor of float64 ready to be used in models, in shape (n, d).
        All methods are generally a lot slower on GPU than on CPU,
        but still faster than having this class on CPU and moving output to GPU.
        """
        self.domain = torch.tensor(domain, dtype=torch.float64, device=device)
        self.device = device
        self.center = (self.domain[:, 1] + self.domain[:, 0]) / 2
        self.amp = (self.domain[:, 1] - self.domain[:, 0]) / 2
        self.length = self.domain[:, 1] - self.domain[:, 0]
        self.uniform_bnrdy_prob = 1 - self.length / self.length.sum()
        self.size = self.domain.shape[0]

    def rand(self, n: int) -> torch.Tensor:
        """
        Return a uniform random sample of size n from the domain.

        Parameters:
        n (int): The number of samples to generate.

        Returns:
        torch.Tensor: A tensor of shape (n, d) with uniform random samples.
        """
        rand = torch.rand(n, self.domain.shape[0], device=self.device) * 2 - 1
        return self.center + self.amp * rand

    def randn(self, n: int, center=None, amp=None) -> torch.Tensor:
        """
        Return a normal random sample of size n from the domain.

        Parameters:
        n (int): The number of samples to generate.
        center (torch.Tensor, optional): The center of the normal distribution.
        amp (torch.Tensor, optional): The amplitude of the normal distribution.

        Returns:
        torch.Tensor: A tensor of shape (n, d) with normal random samples.

        Note:
        Samples can be out of domain, use bound_randn to get in-domain samples.
        """
        if center is None:
            center = self.center
        if amp is None:
            amp = self.amp / 3
        rand = center + amp * torch.randn(n, self.domain.shape[0], device=self.device)
        return rand

    def bound_randn(self, n: int, center=None, amp=None) -> torch.Tensor:
        """
        Return a normal random sample of size n from the domain.
        If out of domain, replace with uniform sample from domain.

        Parameters:
        n (int): The number of samples to generate.
        center (torch.Tensor, optional): The center of the normal distribution.
        amp (torch.Tensor, optional): The amplitude of the normal distribution.

        Returns:
        torch.Tensor: A tensor of shape (n, d) with normal random samples.
        """
        normal = self.randn(n, center, amp)
        uniform = self.rand(n)
        in_domain = torch.logical_and(torch.gt(normal, self.domain[:, 0]), torch.lt(normal, self.domain[:, 1]))
        return torch.where(in_domain, normal, uniform)

    def grid(self, n: int) -> list[torch.Tensor]:
        """
        Return a grid of size n in the domain.

        Parameters:
        n (int): The number of points in each dimension.

        Returns:
        list[torch.Tensor]: A list of tensors representing the grid.

        Note:
        The grid is lexicographically ordered (first dimension changes fastest).
        """
        grids = []
        for i in self.domain:
            grids.append(torch.linspace(i[0], i[1], n, device=self.device, dtype=torch.float64))
        grids.reverse()
        grids = list(torch.meshgrid(grids, indexing='ij'))
        grids.reverse()
        return grids

    def fgrid(self, n: int) -> torch.Tensor:
        """
        Return a flattened grid.

        Parameters:
        n (int): The number of points in each dimension.

        Returns:
        torch.Tensor: A tensor of shape (n^d, d) with the flattened grid.

        Note:
        The grid is lexicographically ordered (first dimension changes fastest).
        """
        return torch.stack(self.grid(n)).reshape(self.size, -1).T

    def regrid(self, grid, n=None, dims=None) -> torch.Tensor:
        """
        Reshape a flattened grid back to its original shape.

        Parameters:
        grid (torch.Tensor): The flattened grid.
        n (int, optional): The number of points in each dimension.
        dims (int, optional): The number of dimensions.

        Returns:
        torch.Tensor: The reshaped grid.
        """
        if n is None:
            n = int(round(grid.shape[0] ** (1 / self.size), 0))
        if dims is None:
            if len(grid.shape) == 1:
                dims = 1
            elif len(grid.shape) == 2:
                dims = grid.shape[1]
            else:
                raise ValueError('grid must be 1D or 2D')
        shape = [n] * self.size
        shape.insert(0, dims)
        if dims == 1:
            return grid.reshape(*shape)
        elif dims >= 2:
            grid = grid.T
            return grid.reshape(*shape)
        else:
            raise ValueError('dims must be >= 1')

    def min(self) -> torch.Tensor:
        """
        Return the minimum values of the domain.

        Returns:
        torch.Tensor: A tensor with the minimum values of the domain.
        """
        return self.domain[:, 0]

    def max(self) -> torch.Tensor:
        """
        Return the maximum values of the domain.

        Returns:
        torch.Tensor: A tensor with the maximum values of the domain.
        """
        return self.domain[:, 1]

    def bndry_rand(self, n) -> torch.Tensor:
        """
        Return a uniform random sample of size n from the boundary of the domain.

        Parameters:
        n (int): The number of samples to generate.

        Returns:
        torch.Tensor: A tensor of shape (n, d) with uniform random samples from the boundary.
        """
        rand = self.rand(n)
        which_dim = self.uniform_bnrdy_prob.multinomial(n, replacement=True)
        which_bndry = torch.randint(0, 2, (n,))  # left(0) or right(1) boundary
        rand[torch.arange(n), which_dim] = self.domain[which_dim, which_bndry]
        return rand

    def select_bndry_rand(self, n, boundary_selection: torch.Tensor) -> torch.Tensor:
        """
        Return a uniform random sample of size n from selected boundaries of the domain.

        Parameters:
        n (int): The number of samples to generate.
        boundary_selection (torch.Tensor): A tensor indicating which boundaries to select.

        Returns:
        torch.Tensor: A tensor of shape (n, d) with uniform random samples from the selected boundaries.

        Note:
        quite slow
        """
        rand = self.rand(n)
        s = boundary_selection.sum(1)/2
        p = s*self.uniform_bnrdy_prob
        p /= p.sum()
        which_dim = p.multinomial(n, replacement=True)
        which_bndry = torch.randint(0, 2, (n,), device=self.device)  # left(0) or right(1) boundary
        which_bndry = torch.where((s==0.5)[which_dim], torch.where(boundary_selection[which_dim, 0] == 1, 0, 1), which_bndry)
        rand[torch.arange(n), which_dim] = self.domain[which_dim, which_bndry]
        return rand


class DataLoader:
    def __init__(self, all_data: torch.Tensor, batch_size:int, device='cpu', output_requires_grad=False, shuffle=True):
        self.all_data = all_data.to(device)
        self.n = all_data.shape[0]
        self.batch_size = batch_size
        self.device = device
        self.output_requires_grad = output_requires_grad
        self.do_shuffle = shuffle
        if self.do_shuffle:
            self.__shuffle()
        self.n_batches = self.n // batch_size
        self.i = 0

    def __shuffle(self):
        self.all_data = self.all_data[torch.randperm(self.n)]

    def __call__(self):
        if self.i == self.n_batches:
            self.i = 0
            if self.do_shuffle:
                self.all_data = self.all_data[torch.randperm(self.n)]
        data = self.all_data[self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        if self.output_requires_grad:
            return data.requires_grad_(True)
        else:
            return data


class timer:
    """Simple timer class"""

    def __init__(self):
        self.t_start = 0
        self.t_end = 0
        self.t_elapsed = 0
        self.running = False

    def start(self):
        self.t_start = time.time()
        self.running = True

    def stop(self):
        self.t_end = time.time()
        self.running = False
        self.t_elapsed += self.t_end - self.t_start

    def read(self):
        if self.running:
            out = self.t_elapsed + (time.time() - self.t_start)
        else:
            out = self.t_elapsed
        print(f'Time elapsed: {format_time(out)}')

    def rr(self):
        # read reset
        self.read()
        self.reset()

    def reset(self):
        self.__init__()


def format_time(t):
    if t < 1e-6:
        return f"{t * 1e9:.2f} ns"
    elif t < 1e-3:
        return f"{t * 1e6:.2f} µs"
    elif t < 1:
        return f"{t * 1e3:.2f} ms"
    elif t < 60:
        return f'{t:.2f} s'
    elif t < 3600:
        return f'{t / 60:.2f} min'
    else:
        return f'{t / 3600:.2f} h'



if __name__ == "__main__":
    # tests but not in test file
    def count_points_on_axes(points: torch.Tensor, tolerance: float = 1e-6) -> list:
        counts = []
        for i in range(points.shape[1]):
            counts.append(torch.sum(torch.abs(points[:, points.shape[1] - i - 1]) <= tolerance).item())
        return counts
    device = get_device('cpu')
    domain = [[0, 1], [0, 1], [0, 1]]
    space = ParameterSpace(domain, device)
    bound_sel = torch.tensor([[1, 1], [1, 1], [0, 0]], device=device)
    tmr = timer(); tmr.start()
    loc = space.select_bndry_rand(10000, bound_sel)
    tmr.stop(); tmr.read()
    # loc = space.bndry_rand(1000)
    print(count_points_on_axes(loc))
    loc = if_tensors_to_numpy(loc)[0]
    if loc.shape[1] == 2:
        plt.scatter(loc[:, 0], loc[:, 1], s=1)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    elif loc.shape[1] == 3:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(loc[:, 0], loc[:, 1], loc[:, 2], s=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()

