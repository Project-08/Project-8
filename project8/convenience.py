# This file contains some convenience functions for the project

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from warnings import warn
import time

def get_device(override: str = None):
    if override is not None:
        return override
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        warn('CUDA/ROCm not available, Using CPU.')
    return device

def plot_2d(x, y, f, title, fig_id=0, filename=None):
    if filename is None:
        filename = title + '.png'
    plt.figure(fig_id)
    plt.clf()
    domain = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(f, extent=domain, origin='lower')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(title)
    if domain[1] - domain[0] <1.1* (domain[3] - domain[2]):
        plt.colorbar(orientation='vertical')
    else:
        plt.colorbar(orientation='horizontal')
    plt.savefig('plots/'+filename)

def plot_plane(x, y, f, title, fig_id=0, filename=None):
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
    plt.savefig('plots/' + filename)

class float_parameter_space:
    def __init__(self, domain: iter, device='cpu'):
        # domain is a 2D tensor of floats, for constant parameters, use [a, a]
        # most methods output a tensor of float64 ready to be used in models, in shape (n, d)
        # all methods are generally a lot slower on gpu than on cpu,
        # but still faster than having this class on cpu and moving output to gpu
        self.domain = torch.tensor(domain, dtype=torch.float64, device=device)
        self.device = device
        self.center = (self.domain[:,1] + self.domain[:,0]) / 2
        self.amp = (self.domain[:,1] - self.domain[:,0]) / 2
        self.size = self.domain.shape[0]

    def rand(self, n: int) -> torch.Tensor:
        # return a uniform random sample of size n from the domain
        # fast
        rand = torch.rand(n, self.domain.shape[0], device=self.device)*2-1
        return self.center + self.amp * rand

    def randn(self, n: int, center=None, amp=None) -> torch.Tensor:
        # return a normal random sample of size n from the domain
        # !!!! can be out of domain, use bound_randn to get in-domain samples
        # fast
        if center is None:
            center = self.center
        if amp is None:
            amp = self.amp / 3
        rand = center + amp * torch.randn(n, self.domain.shape[0], device=self.device)
        return rand

    def bound_randn(self, n: int, center=None, amp=None) -> torch.Tensor:
        # return a normal random sample of size n from the domain
        # if out of domain, replace with uniform sample from domain
        # slow
        normal = self.randn(n, center, amp)
        uniform = self.rand(n)
        in_domain = torch.logical_and(torch.gt(normal, self.domain[:,0]), torch.lt(normal, self.domain[:,1]))
        return torch.where(in_domain, normal, uniform)

    def grid(self, n: int) -> list[torch.Tensor]:
        # return a grid of size n in the domain
        # lexicographically ordered (first dimension changes fastest)
        # returns a tensor of shape (d, [n]*d)
        # slow
        grids = []
        for i in self.domain:
            grids.append(torch.linspace(i[0], i[1], n, device=self.device, dtype=torch.float64))
        grids.reverse()
        grids = list(torch.meshgrid(grids, indexing='ij'))
        grids.reverse()
        # grids = [x, y, ..., t]
        return grids

    def fgrid(self, n: int) -> torch.Tensor:
        # flattened grid
        # returns a tensor of shape (n^d, d)
        # lexicographically ordered (first dimension changes fastest)
        # slow
        return torch.stack(self.grid(n)).reshape(self.size, -1).T

    def regrid(self, grid, n=None, dims=None) -> torch.Tensor:
        if n is None:
            n = int(round(grid.shape[0] ** (1 / self.size),0))
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
        return self.domain[:,0]

    def max(self) -> torch.Tensor:
        return self.domain[:,1]

    def bndry_rand(self, n) -> torch.Tensor:
        # return a uniform* random sample of size n from the boundary of the domain
        # *each boundary has same chance of getting chosen, regardless of size, but after that uniform random
        # slow +- 10ms
        rand = self.rand(n)
        which_dim = torch.randint(0, self.size, (n,))
        which_bndry = torch.randint(0, 2, (n,))
        rand[torch.arange(n), which_dim] = self.domain[which_dim, which_bndry]
        return rand

class timer:
    def __init__(self):
        self.t_init = time.time()
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
        return f'{t/60:.2f} min'
    else:
        return f'{t/3600:.2f} h'

if __name__ == "__main__":
    # device = get_device()
    domain = [[-1, 4], [-1, 1]]
    space = float_parameter_space(domain, 'cpu')
    loc = space.bndry_rand(200).detach().to('cpu')
    plt.scatter(loc[:,0], loc[:,1], s=1)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()