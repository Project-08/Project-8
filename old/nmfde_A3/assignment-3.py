import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
import scipy.sparse.linalg as la

plot = True
# parameters
LeftX = -1.0
RightX = 2.0
LeftY = -1.0
RightY = 1.0
Nx = 200  # number of intervals in x-direction
Ny = 200  # number of intervals in y-direction
dx = (RightX - LeftX) / Nx  # grid step in x-direction
dy = (RightY - LeftY) / Ny  # grid step in y-direction
# xgrid = np.linspace(LeftX, RightX, Nx+1)
# ygrid = np.linspace(LeftY, RightY, Ny+1)
# x,y = np.meshgrid(xgrid[1:-1], ygrid[1:-1])
x, y = np.mgrid[LeftX + dx:RightX:dx, LeftY + dy:RightY:dy]
x = x.transpose()
y = y.transpose()
if plot and Nx < 5:
    print(x)
    print(y)


def assemble_D(n, h=1.0):
    d0 = np.repeat(-1, n - 1)
    d1 = np.repeat(1, n - 1)
    D = sp.diags([d0, d1], [-1, 0], shape=(n, n - 1))
    return D / h


def FD_laplacian_2d(Nx, Ny, dx, dy):
    Dx = assemble_D(Nx, dx)
    Lxx = Dx.transpose().dot(Dx)
    Ix = sp.eye(Nx - 1)
    Dy = assemble_D(Ny, dy)
    Lyy = Dy.transpose().dot(Dy)
    Iy = sp.eye(Ny - 1)
    A = sp.kron(Iy, Lxx) + sp.kron(Lyy, Ix)
    return A


def sourcefunc(x, y, alpha=40):
    sum = 0
    for i in range(1, 10):
        for j in range(1, 5):
            sum += np.exp(-alpha * (x - i) ** 2 - alpha * (y - j) ** 2)
    return sum


def sourcefunc_const(x, y, const=-3):
    return x * const


def source_2(x, y):
    return np.exp(-20 * (x - 0.5) ** 2 - 20 * (y - 0.5) ** 2) + 0.1


def source3(x, y):
    return np.sin(4 * x * y)


f = source3(x, y)
flx = np.reshape(f, (Nx - 1) * (Ny - 1), order='C')
A = FD_laplacian_2d(Nx, Ny, dx, dy)
u = la.spsolve(A, flx)
# reshaping the solution vector into 2D array
uArr = np.reshape(u, (Ny - 1, Nx - 1), order='C')
uArr = np.pad(uArr, ((1, 1), (1, 1)), 'constant', constant_values=0)

# visualizing the matrix
if plot:
    if Nx < 5:
        print(A.toarray())
    plt.figure(0)
    plt.spy(A)
    plt.savefig('plots/sparsity.png')
    # plt.show()
# visualizing the source function
if plot:
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.imshow(f, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    # additional commands to make your plot look correct
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Source function')
    plt.savefig('plots/source.png')
    # plt.show()

# visualizing the solution
if plot:
    plt.figure(2)
    plt.clf()
    plt.imshow(uArr, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    # additional commands to make your plot look correct
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution')
    plt.savefig('plots/solution.png')
    # plt.show()


# FVM

def coeffK1(x, y):
    return x * 0 + 1


def coeffK2(x, y):
    return 1 + 0.1 * (x + y + x * y)


def create2DLFVM(Nx, Ny, dx, dy, coeffFun):
    # main diagonal
    diag = np.zeros((Nx - 1) * (Ny - 1))
    upper = np.zeros((Nx - 1) * (Ny - 1) - 1)
    lower = np.zeros((Nx - 1) * (Ny - 1) - 1)
    upper_upper = np.zeros((Nx - 1) * (Ny - 1) - Nx + 1)
    lower_lower = np.zeros((Nx - 1) * (Ny - 1) - Nx + 1)
    if Nx < 5:
        print(len(diag), len(upper), len(lower), len(upper_upper), len(lower_lower))
    for j in range(1, Ny):
        for i in range(1, Nx):
            idx = (j - 1) * (Nx - 1) + i - 1
            if Nx < 5:
                print('i =', i, 'j =', j, 'idx =', idx)
                print('i*dx =', i * dx, 'j*dy =', j * dy)
            x = i * dx
            y = j * dy
            diag[idx] = ((coeffFun(x - dx / 2, y) / dx ** 2) + (coeffFun(x, y - dy / 2)) / dy ** 2) + (
                        (coeffFun(x + dx / 2, y) / dx ** 2) + (coeffFun(x, y + dy / 2)) / dy ** 2)
            if i < Nx - 1:
                upper[idx] = -coeffFun(x + dx / 2, y) / dx ** 2
            if i > 1:
                lower[idx - 1] = -coeffFun(x - dx / 2, y) / dx ** 2
            if j < Ny - 1:
                upper_upper[idx] = -coeffFun(x, y + dy / 2) / dy ** 2
            if j > 1:
                lower_lower[idx - Nx + 1] = -coeffFun(x, y - dy / 2) / dy ** 2
    A = sp.diags([lower_lower, lower, diag, upper, upper_upper], [-(Nx - 1), -1, 0, 1, Nx - 1],
                 shape=((Nx - 1) * (Ny - 1), (Nx - 1) * (Ny - 1)), format='csc')
    return A


f = sourcefunc(x, y)
flx = np.reshape(f, (Nx - 1) * (Ny - 1), order='C')
A = create2DLFVM(Nx, Ny, dx, dy, coeffK2)
u = la.spsolve(A, flx)
uArr = np.reshape(u, (Ny - 1, Nx - 1), order='C')
uArr = np.pad(uArr, ((1, 1), (1, 1)), 'constant', constant_values=0)

if plot:
    if Nx < 5:
        print(A.toarray())
    plt.figure(3)
    plt.spy(A)
    plt.savefig('plots/sparsity_fvm.png')
    # plt.show()

if plot:
    plt.figure(4)
    plt.clf()
    plt.subplot(1, 2, 1)
    plt.imshow(coeffK1(x, y), extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.title('k1')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.subplot(1, 2, 2)
    plt.imshow(coeffK2(x, y), extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.title('k2')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plots/k_functions.png', bbox_inches='tight')

# visualizing the solution
if plot:
    plt.figure(5)
    plt.clf()
    plt.imshow(uArr, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    plt.colorbar()
    # additional commands to make your plot look correct
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Solution FVM')
    plt.savefig('plots/solution_fvm.png')
    # plt.show()
