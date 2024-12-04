"""This file contains some convenience functions for the project"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore
from warnings import warn
import time
import os
import logging
from typing import Optional, List, Iterable, Any, Union, TypedDict, Callable, \
    Dict

plot_folder = os.path.join(
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.abspath(__file__)
            )
        )
    ), 'plots')
logging.info(f'plot_folder: {plot_folder}')


def get_device(override: str = '') -> str:
    if override != '':
        return override
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warn('CUDA/ROCm not available, Using CPU.')
    return device


def if_tensors_to_numpy(*args: Any) -> List[np.ndarray[Any, Any]]:
    out = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            out.append(arg.detach().to('cpu').numpy())
        elif isinstance(arg, np.ndarray):
            out.append(arg)
        else:
            out.append(np.array(arg))
    return out


def plot_2d(x: Union[torch.Tensor, np.ndarray[Any, Any]],
            y: Union[torch.Tensor, np.ndarray[Any, Any]],
            f: Union[torch.Tensor, np.ndarray[Any, Any]], title: str,
            fig_id: int = 0,
            filename: str = '') -> None:
    x, y, f = if_tensors_to_numpy(x, y, f)
    if filename == '':
        filename = title + '.png'
    plt.figure(fig_id)
    plt.clf()
    domain = (x.min(), x.max(), y.min(), y.max())
    plt.imshow(f, extent=domain, origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if domain[1] - domain[0] < 1.1 * (domain[3] - domain[2]):
        plt.colorbar(orientation='vertical')
    else:
        plt.colorbar(orientation='horizontal')
    plt.savefig(plot_folder + '/' + filename)


def plot_plane(x: Union[torch.Tensor, np.ndarray[Any, Any]],
               y: Union[torch.Tensor, np.ndarray[Any, Any]],
               f: Union[torch.Tensor, np.ndarray[Any, Any]], title: str,
               fig_id: int = 0,
               filename: str = '') -> None:
    x, y, f = if_tensors_to_numpy(x, y, f)
    if filename is None:
        filename = title + '.png'
    fig = plt.figure(fig_id)
    plt.clf()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, f, cmap='viridis')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f')
    ax.set_title(title)
    plt.savefig(plot_folder + '/' + filename)


class ParameterSpace:
    def __init__(self, domain: Iterable[Iterable[float]],
                 device: str = 'cpu') -> None:
        self.domain = torch.tensor(domain, dtype=torch.float64, device=device)
        self.device = device
        self.center = (self.domain[:, 1] + self.domain[:, 0]) / 2
        self.amp = (self.domain[:, 1] - self.domain[:, 0]) / 2
        self.length = self.domain[:, 1] - self.domain[:, 0]
        self.uniform_bnrdy_prob = 1 - self.length / self.length.sum()
        self.size = self.domain.shape[0]

    def rand(self, n: int) -> torch.Tensor:
        rand = torch.rand(n, self.domain.shape[0], device=self.device) * 2 - 1
        return self.center + self.amp * rand

    def randn(self, n: int, center: Optional[torch.Tensor] = None,
              amp: Optional[torch.Tensor] = None) -> torch.Tensor:
        if center is None:
            center = self.center
        if amp is None:
            amp = self.amp / 3
        rand = center + amp * torch.randn(n, self.domain.shape[0],
                                          device=self.device)
        return rand

    def bound_randn(self, n: int, center: Optional[torch.Tensor] = None,
                    amp: Optional[torch.Tensor] = None) -> torch.Tensor:
        normal = self.randn(n, center, amp)
        uniform = self.rand(n)
        in_domain = torch.logical_and(torch.gt(normal, self.domain[:, 0]),
                                      torch.lt(normal, self.domain[:, 1]))
        return torch.where(in_domain, normal, uniform)

    def grid(self, n: int) -> List[torch.Tensor]:
        grids = []
        for i in self.domain:
            grids.append(torch.linspace(i[0], i[1], n, device=self.device,
                                        dtype=torch.float64))
        grids.reverse()
        grids = list(torch.meshgrid(grids, indexing='ij'))
        grids.reverse()
        return grids

    def fgrid(self, n: int) -> torch.Tensor:
        return torch.stack(self.grid(n)).reshape(self.size, -1).T

    def regrid(self, grid: torch.Tensor, n: Optional[int] = None,
               dims: Optional[int] = None) -> torch.Tensor:
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
        return self.domain[:, 0]

    def max(self) -> torch.Tensor:
        return self.domain[:, 1]

    def bndry_rand(self, n: int) -> torch.Tensor:
        rand = self.rand(n)
        which_dim = self.uniform_bnrdy_prob.multinomial(n, replacement=True)
        which_bndry = torch.randint(0, 2, (n,))
        rand[torch.arange(n), which_dim] = self.domain[which_dim, which_bndry]
        return rand

    def select_bndry_rand(self, n: int,
                          boundary_selection: torch.Tensor) -> torch.Tensor:
        rand = self.rand(n)
        s = boundary_selection.sum(1) / 2
        p = s * self.uniform_bnrdy_prob
        p /= p.sum()
        which_dim = p.multinomial(n, replacement=True)
        which_bndry = torch.randint(0, 2, (n,), device=self.device)
        which_bndry = torch.where((s == 0.5)[which_dim], torch.where(
            boundary_selection[which_dim, 0] == 1, 0, 1), which_bndry)
        rand[torch.arange(n), which_dim] = self.domain[which_dim, which_bndry]
        return rand


class DataLoader:
    """
    DataLoader for training neural networks.
    Device defaults to device of data
    """

    def __init__(self, all_data: torch.Tensor, batch_size: int,
                 output_requires_grad: bool = False,
                 shuffle: bool = True,
                 device: Optional[str | torch.device] = None) -> None:
        if device is not None:
            self.all_data = all_data.to(device)
            self.device = device
        else:
            self.all_data = all_data
            self.device = str(self.all_data.device)
        self.n = all_data.shape[0]
        self.batch_size = batch_size
        self.output_requires_grad = output_requires_grad
        self.do_shuffle = shuffle
        if self.do_shuffle:
            self.__shuffle()
        self.n_batches = self.n // batch_size
        self.i = 0

    def __shuffle(self) -> None:
        self.all_data = self.all_data[torch.randperm(self.n)]

    def __call__(self) -> torch.Tensor:
        if self.i == self.n_batches:
            self.i = 0
            if self.do_shuffle:
                self.__shuffle()
        data = self.all_data[
               self.i * self.batch_size:(self.i + 1) * self.batch_size]
        self.i += 1
        return data.requires_grad_(self.output_requires_grad)


class Timer:
    """Simple timer class"""

    def __init__(self) -> None:
        self.t_start: float = 0
        self.t_end: float = 0
        self.t_elapsed: float = 0
        self.running: bool = False

    def start(self) -> None:
        self.t_start = time.time()
        self.running = True

    def stop(self) -> None:
        self.t_end = time.time()
        self.running = False
        self.t_elapsed += self.t_end - self.t_start

    def elapsed(self) -> float:
        if self.running:
            return self.t_elapsed + (time.time() - self.t_start)
        else:
            return self.t_elapsed

    def read(self) -> None:
        print(f'Time elapsed: {format_time(self.elapsed())}')

    def rr(self) -> None:
        # read reset
        self.read()
        self.reset()

    def reset(self) -> None:
        self.t_start = 0
        self.t_end = 0
        self.t_elapsed = 0
        self.running = False


def format_time(t: float) -> str:
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


class Params(TypedDict, total=False):
    device: torch.device | str
    # training params
    n_epochs: int
    optimizer: Any  # torch.optim.Optimizer. typing doesnt like optim.
    lr: float
    loss_fns: list[Callable[[Any], torch.Tensor]]
    loss_weights: list[float]
    loss_fn_data: list[torch.Tensor]
    loss_fn_batch_sizes: list[int]
    # model params
    model_constructor: str
    weight_init_fn: Callable[
                     [torch.Tensor, Any], torch.Tensor
                 ] | Callable[
                     [torch.Tensor], torch.Tensor
                 ]
    bias_init_fn: Callable[
                   [torch.Tensor, Any], torch.Tensor
               ] | Callable[
                   [torch.Tensor], torch.Tensor
               ]
    weight_init_kwargs: Optional[Dict[str, Any]]
    bias_init_kwargs: Optional[Dict[str, Any]]
    # rectangular_fnn
    n_in: int
    n_out: int
    width: int
    depth: int
    act_fn: torch.nn.Module
    # drm
    n_blocks: int
    n_linear_drm: int
