import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la
from multiprocessing import Pool
from typing import List, Any, Callable


def fd_laplacian_9_point_nd(steps: int,
                            stepsize: float,
                            dimensions: int) -> Any:
    diagonal = np.ones(steps)/stepsize
    offsets: List[int] = [0, -1]
    d = sp.diags([diagonal, -diagonal],
                 offsets=offsets,
                 shape=(steps, steps-1))
    L = d.T@d

    sub_matrices = list()
    for dim in range(dimensions):
        I1 = sp.eye(int((steps-1)**dim))
        I2 = sp.eye(int((steps-1)**(dimensions-dim-1)))
        sub_matrices.append(sp.kron(sp.kron(I1, L), I2))
    return np.sum(sub_matrices)


def fd_laplacian_13_point_nd(steps: int, stepsize: float) -> Any:
    diagonal1 = np.ones(steps-1)/(stepsize**2)*2.5
    diagonal2 = np.ones(steps-2)/(stepsize**2)*-(4/3)
    diagonal3 = np.ones(steps-3)/(stepsize**2)*(1/12)
    diagonal1[0] -= diagonal3[0]
    diagonal1[-1] -= diagonal3[-1]
    offsets: List[int] = [-2, -1, 0, 1, 2]
    L = sp.diags(
                 [diagonal3, diagonal2, diagonal1, diagonal2, diagonal3],
                 offsets=offsets,
                 shape=(steps-1, steps-1)
                 )
    dimensions = 3
    sub_matrices = list()
    for dim in range(dimensions):
        I1 = sp.eye(int((steps-1)**dim))
        I2 = sp.eye(int((steps-1)**(dimensions-dim-1)))
        sub_matrices.append(sp.kron(sp.kron(I1, L), I2))
    return np.sum(sub_matrices)


def _source_func(x: float, y: float, z: float) -> np.float64:
    return np.float64(-np.pi*np.pi*x*y*np.sin(np.pi*z))


def _get_index(x: int, y: int, z: int, steps: int) -> int:
    return (x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)


def _get_9_point_source_func(
        func: Callable[[float, float, float], np.float64],
        steps: int,
        stepsize: float
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    result = np.zeros((steps-1)**3)
    for z in range(1, steps):
        for y in range(1, steps):
            for x in range(1, steps):
                result[_get_index(x, y, z, steps)] =\
                    func(x*stepsize, y*stepsize, z*stepsize)
                # Boundary Condition:
                if (y == (steps-1)):
                    result[_get_index(x, y, z, steps)]\
                        += x*stepsize*np.sin(np.pi*z*stepsize)/(stepsize**2)
                if (x == (steps-1)):
                    result[_get_index(x, y, z, steps)]\
                        += y*stepsize*np.sin(np.pi*z*stepsize)/(stepsize**2)
    return result


def _get_13_point_source_func(
        func: Callable[[float, float, float], np.float64],
        steps: int,
        stepsize: float
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    result = np.zeros((steps-1)**3)
    for z in range(1, steps):
        for y in range(1, steps):
            for x in range(1, steps):
                result[_get_index(x, y, z, steps)] =\
                    func(x*stepsize,
                         y*stepsize,
                         z*stepsize
                         )
                # Boundary Condition:
                if (y == (steps-2)):
                    result[_get_index(x, y, z, steps)]\
                        -= (1/12)*x*stepsize*np.sin(np.pi*z*stepsize)\
                        / (stepsize**2)
                if (y == (steps-1)):
                    result[_get_index(x, y, z, steps)]\
                        += (4/3-0.16)*x*stepsize*np.sin(np.pi*z*stepsize)\
                        / (stepsize**2)\
                        + (1/12) * func(x*stepsize,
                                        (y+1)*stepsize,
                                        z*stepsize)
                if (y == 1):
                    result[_get_index(x, y, z, steps)]\
                        += (1/12) * func(x*stepsize,
                                         (y-1)*stepsize,
                                         z*stepsize)

                if (x == (steps-2)):
                    result[_get_index(x, y, z, steps)]\
                        -= (1/12)*y*stepsize*np.sin(np.pi*z*stepsize)\
                        / (stepsize**2)
                if (x == (steps-1)):
                    result[_get_index(x, y, z, steps)]\
                        + (4/3-0.16)*y*stepsize*np.sin(np.pi*z*stepsize)\
                        / (stepsize**2)\
                        + (1/12) * func((x+1)*stepsize,
                                        y*stepsize,
                                        z*stepsize)
                if (x == 1):
                    result[_get_index(x, y, z, steps)]\
                        += (1/12) * func((x-1)*stepsize,
                                         y*stepsize,
                                         z*stepsize)

                if (z == (steps-1)):
                    result[_get_index(x, y, z, steps)]\
                        += (1/12) * func(x*stepsize,
                                         y*stepsize,
                                         (z+1)*stepsize)
                if (z == 1):
                    result[_get_index(x, y, z, steps)]\
                        += (1/12) * func(x*stepsize,
                                         y*stepsize,
                                         (z-1)*stepsize)

    return result


def _get_3d_analytical(
        steps: int,
        stepsize: float
        ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

    result = np.zeros((steps-1)**3)
    for z in range(1, steps):
        for y in range(1, steps):
            for x in range(1, steps):
                result[_get_index(x, y, z, steps)] =\
                    stepsize*stepsize*x*y*np.sin(np.pi*z*stepsize)
    return result


def _get_error_9(size: int) -> np.float64:
    n = size
    u_num = la.spsolve(
        fd_laplacian_9_point_nd(n, 1/n, 3),
        _get_9_point_source_func(_source_func, n, 1/n)
        )
    u_exact = _get_3d_analytical(n, 1/n)
    squares = (u_num-u_exact)**2
    mean = np.mean(squares)
    root = np.sqrt(mean)
    return np.float64(root)


def _get_error_13(size: int) -> np.float64:
    n = size
    u_num = la.spsolve(
        fd_laplacian_13_point_nd(n, 1/n),
        _get_13_point_source_func(_source_func, n, 1/n)
        )
    u_exact = _get_3d_analytical(n, 1/n)
    squares = (u_num-u_exact)**2
    mean = np.mean(squares)
    root = np.sqrt(mean)
    return np.float64(root)


def _generate_plot() -> None:
    steps_array = np.array([4])
    with Pool() as pool:
        RMS_13 = pool.map(_get_error_13, steps_array)
    with Pool() as pool:
        RMS_9 = pool.map(_get_error_9, steps_array)
    plt.semilogy(steps_array, RMS_9, label="RMS_9")
    plt.semilogy(steps_array, RMS_13, label="RMS_13")
    plt.legend()
    plt.show()
    plt.savefig("3D Error until N=50")
