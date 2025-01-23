from project8.neural_network import models
from project8.neural_network import differential_operators as diffop
import torch
from typing import Callable


def sinxy_source(input: torch.Tensor) -> torch.Tensor:
    return -torch.sin(12 * input[:, 0] * input[:, 1]).unsqueeze(1)


def nmfde_a3_source(input: torch.Tensor) -> torch.Tensor:
    alpha = 40
    x = input[:, 0].unsqueeze(1)
    y = input[:, 1].unsqueeze(1)
    sum = torch.zeros_like(x)
    for i in range(1, 10):
        for j in range(1, 5):
            sum += torch.exp(-alpha * (x - i) ** 2 - alpha * (y - j) ** 2)
    return sum


def nmfde_a4_wave(input: torch.Tensor) -> torch.Tensor:
    Lx = 10
    Ly = 5
    alpha = 40
    omega = 4 * torch.pi
    f = torch.zeros_like(input[:, 0])
    a = [
        [0.25 * Lx, 0.25 * Ly],
        [0.25 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.75 * Ly],
        [0.75 * Lx, 0.25 * Ly]
    ]
    for i in a:
        f += torch.exp(
            -alpha * (input[:, 0] - i[0]) ** 2 - alpha * (
                    input[:, 1] - i[1]) ** 2)
    return (f * torch.sin(omega * input[:, 2])).unsqueeze(1)


def zero_source(input: torch.Tensor) -> torch.Tensor:
    return torch.zeros_like(input[:, 0]).unsqueeze(1)


class PINN:
    class laplacian:
        # -laplacian(u) = f
        # residual = laplacian(u) + f
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            f = self.source(model.input)
            laplace = diffop.laplacian(model, time_dependent=False)
            return (laplace + f).pow(2).mean()

    class wave:
        # u_tt = laplacian(u) + f
        # residual = u_tt - laplacian(u) - f
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            t_dim_id = model.input.shape[1] - 1
            f = self.source(model.input)
            residual = diffop.partial_derivative(model, t_dim_id,
                                                 t_dim_id) - diffop.laplacian(
                model,
                time_dependent=True) - f
            return residual.pow(2).mean()

    class navier_stokes_2d:
        """
        2d incompressible Navier-Stokes equation
        velocity components = [U, V]
        pressure = P
        density = rho
        viscosity = v
        U_t + U*U_x + V*U_y = (1/rho) * (-P_x + v * laplacian(U))
        V_t + U*V_x + V*V_y = (1/rho) * (-P_y + v * laplacian(V))
        U_x + V_y = 0
        """
        def __init__(self,
                     viscosity: float = 1e-3,
                     density: float = 1.0,
                     a: float = 1.0,
                     b: float = 1.0,
                     c: float = 1.0
                     ) -> None:
            self.v = viscosity
            self.rho = density
            self.a = a
            self.b = b
            self.c = c

        def __call__(self, model: models.NN) -> torch.Tensor:
            u = model.output[:, 0].unsqueeze(1)
            v = model.output[:, 1].unsqueeze(1)
            # u: 0, v: 1, p: 2
            u_x = diffop.partial_derivative(model, 0, out_dim_index=0)
            u_y = diffop.partial_derivative(model, 1, out_dim_index=0)
            u_t = diffop.partial_derivative(model, 2, out_dim_index=0)
            v_x = diffop.partial_derivative(model, 0, out_dim_index=1)
            v_y = diffop.partial_derivative(model, 1, out_dim_index=1)
            v_t = diffop.partial_derivative(model, 2, out_dim_index=1)
            p_x = diffop.partial_derivative(model, 0, out_dim_index=2)
            p_y = diffop.partial_derivative(model, 1, out_dim_index=2)
            laplacian_u = diffop.laplacian(model, out_dim_index=0, time_dependent=True)
            laplacian_v = diffop.laplacian(model, out_dim_index=1, time_dependent=True)
            u_residual = u_t + u * u_x + v * u_y - (1 / self.rho) * (-p_x + self.v * laplacian_u)
            v_residual = v_t + u * v_x + v * v_y - (1 / self.rho) * (-p_y + self.v * laplacian_v)
            continuity = u_x + v_y
            return self.a * u_residual.pow(2).mean() \
                + self.b * v_residual.pow(2).mean() \
                + self.c *  continuity.pow(2).mean()





class DRM:
    class laplacian:
        # -laplacian(u) = f
        # Minimize 0.5 * ||grad(u)||^2 - f * u
        def __init__(self,
                     source: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.source = source

        def __call__(self, model: models.NN) -> torch.Tensor:
            grad = diffop.gradient(model)
            f = self.source(model.input)
            return torch.mean(
                0.5 * torch.sum(grad.pow(2), 1).unsqueeze(
                    1) - f * model.output)


class General:
    @staticmethod
    def dirichlet_bc(model: models.NN) -> torch.Tensor:
        return model.output.pow(2).mean()

    class fit_to_fn:
        def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> None:
            self.fn = fn

        def __call__(self, model: models.NN) -> torch.Tensor:
            return (model.output - self.fn(model.input)).pow(2).mean()


class NavierStokes2D:
    """
    Test case 2D-2 from
    https://wwwold.mathematik.tu-dortmund.de/lsiii/cms/papers/SchaeferTurek1996.pdf
    with zero pressure outlet condition
    velocity components = [U, V]
    pressure = P
    density = rho
    viscosity = v
    U_t + U*U_x + V*U_y = (1/rho) * (-P_x + v * laplacian(U))
    V_t + U*V_x + V*V_y = (1/rho) * (-P_y + v * laplacian(V))
    U_x + V_y = 0
                                                        changes
    U(x, 0, t) = U(x, h, t) = 0
    V(x, 0, t) = V(x, h, t) = 0
    V(0, y, t) = 0                                      also neumann 0 for P on inlet
    U(0, y, t) = 4 * U_m * y(h-y) / h^2
    P(L, y, t) = 0                                      0 neumann on u v and p on outlet
    U(x, y, t) = V(x, y, t) = 0 | x,y,t on cylinder boundary
    U(x, y, 0) = V(x, y, 0) = P(x, y, 0) = 0            all can have own const IC (default 0)
    x = [0, L], y = [0, h], t = [0, T]
    """

    @staticmethod
    def cylinder_bndry_data(
            n: int,
            cx: float = 0.2,
            cy: float = 0.2,
            r: float = 0.05,
            t_range: tuple[float,float] = (0, 8)
    ) -> torch.Tensor:
        theta = torch.rand(n, dtype=torch.float64) * 2 * torch.pi
        x = cx + r * torch.cos(theta)
        y = cy + r * torch.sin(theta)
        t = torch.rand(n, dtype=torch.float64) * (t_range[1] - t_range[0]) + t_range[0]
        points = torch.stack((x, y, t), dim=1)
        return points

    class inlet_bc:
        def __init__(self, U_m: float = 1.5, h: float = 0.41):
            self.U_m = U_m
            self.h = h

        def __call__(self, model: models.NN) -> torch.Tensor:
            U = model.output[:, 0]
            V = model.output[:, 1]
            y = model.input[:, 1]
            P_x = diffop.partial_derivative(model, 0, out_dim_index=2)
            return (U - 4 * self.U_m * y * (self.h - y) / self.h ** 2).pow(2).mean()\
                + V.pow(2).mean()\
                + P_x.pow(2).mean()

    @staticmethod
    def outlet_bc(model: models.NN) -> torch.Tensor:
        # was p=0, now P_x = U_x = V_x = 0
        return diffop.gradient(model, 2)[:, 0].pow(2).mean()\
                + diffop.gradient(model, 1)[:, 0].pow(2).mean()\
                + diffop.gradient(model, 0)[:, 0].pow(2).mean()

    @staticmethod
    def cylinder_bc(model: models.NN) -> torch.Tensor:
        U = model.output[:, 0]
        V = model.output[:, 1]
        return U.pow(2).mean() + V.pow(2).mean()

    @staticmethod
    def top_bottom_bc(model: models.NN) -> torch.Tensor:
        P_y = diffop.partial_derivative(model, 1, out_dim_index=2)
        return model.output[:, 1].pow(2).mean() + 0.1 * model.output[:, 0].pow(2).mean() + P_y.pow(2).mean()

    class initial_condition:
        # was all 0
        def __init__(self, device: str, U0: float = 0, V0: float = 0.0, P0: float = 0.0):
            self.ic = torch.tensor([[U0, V0, P0]], dtype=torch.float64, device=device)

        def __call__(self, model: models.NN) -> torch.Tensor:
            return (model.output - self.ic).pow(2).mean()


class Problem1:
    """
    Problem 1 from https://doi.org/10.1016/j.apm.2011.11.078
    -laplacian(u) = f
    -f(x, y) = sin(pi*x) * sin(pi*y)
    u(0, y) = u(1, y) = u(x, 0) = u(x, 1) = 0

    u(x, y) = (-1/(2*pi^2)) * sin(pi*x) * sin(pi*y)
    (x,y) = [0,1]x[0,1]
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        return -(torch.sin(torch.pi * input[:, 0]) * torch.sin(
            torch.pi * input[:, 1])).unsqueeze(1)

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        return ((-1 / (2 * torch.pi ** 2)) * torch.sin(
            torch.pi * input[:, 0]) * torch.sin(
            torch.pi * input[:, 1])).unsqueeze(1)

    # dirichlet bc


class Problem2:
    """
    Equation 25 and 26 from https://doi.org/10.1016/j.procs.2010.04.041
    -laplacian(u) = f
    -f(x, y) = (x^2 + y^2) * e^(x*y)
    u(a, y) = u(x, a) = u(b, y) = u(x, b) = e^(x*y)
    (x,y) = [a,b]x[a,b]

    u(x, y) = e^(x*y)
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = -(input[:, 0] ** 2 + input[:, 1] ** 2) * torch.exp(
            input[:, 0] * input[:, 1]).unsqueeze(1)
        return out

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        return (torch.exp(input[:, 0] * input[:, 1])).unsqueeze(1)

    @staticmethod
    def bc(model: models.NN) -> torch.Tensor:
        residual = model.output - Problem2.exact(model.input)
        return residual.pow(2).mean()


class Problem3:
    """
    Example 3 from
    https://www.researchgate.net/publication
    /266007367_An_Efficient_Direct_Method_to_Solve_the_Three_Dimensional_Poisson's_Equation
    -laplacian(u) = f
    -f(x, y, z) = 2 * (xy + xz + yz)
    u(0, y, z) = u(x, 0, z) = u(x, y, 0) =  0
    u(1, y, z) = yz(1 + y + z)
    u(x, 1, z) = xz(x + 1 + z)
    u(x, y, 1) = xy(x + y + 1)
    (implemented as u(1, y, z) = u(x, 1, z) = u(x, y, 1) = xyz(x + y + z))
    (x,y,z) = [0,1]x[0,1]x[0,1]

    u(x, y, z) = xyz(x + y + z)
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        return -(2 * (
            input[:, 0] * input[:, 1]
            + input[:, 0] * input[:, 2]
            + input[:, 1] * input[:, 2])
        ).unsqueeze(1)

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        return (input[:, 0] * input[:, 1] * input[:, 2] * (
                input[:, 0] + input[:, 1] + input[:, 2])).unsqueeze(1)

    # dirichlet bc

    @staticmethod
    def bc(model: models.NN) -> torch.Tensor:
        residual = model.output - Problem3.exact(model.input)
        return residual.pow(2).mean()


class Problem4:
    """
    Example 4 from
    https://www.researchgate.net/publication
    /266007367_An_Efficient_Direct_Method_to_Solve_the_Three_Dimensional_Poisson's_Equation
    -laplacian(u) = f
    -f(x, y, z) = 6
    u(0, y, z) = y^2 + z^2
    u(x, 0, z) = x^2 + z^2
    u(x, y, 0) = x^2 + y^2
    u(1, y, z) = 1 + y^2 + z^2
    u(x, 1, z) = 1 + x^2 + z^2
    u(x, y, 1) = 1 + x^2 + y^2
    (implemented as u(0, y, z) = u(x, 0, z) = u(x, y, 0) = u(1, y, z) = u(x,
    1, z) = u(x, y, 1) = x^2 + y^2 + z^2)
    (x,y,z) = [0,1]x[0,1]x[0,1]

    u(x, y, z) = x^2 + y^2 + z^2
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        return ((input[:, 0] * 0) - 6).unsqueeze(1)

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = (
                input[:, 0] ** 2 + input[:, 1] ** 2 + input[:, 2] ** 2
        ).unsqueeze(1)
        return out

    @staticmethod
    def bc(model: models.NN) -> torch.Tensor:
        residual = model.output - Problem4.exact(model.input)
        return residual.pow(2).mean()


class Problem5:
    """
    Example 5 from
    https://www.researchgate.net/publication
    /266007367_An_Efficient_Direct_Method_to_Solve_the_Three_Dimensional_Poisson's_Equation
    -laplacian(u) = f
    -f(x, y, z) = -pi^2 * x * y * sin(pi * z)
    u(0, x, z) = u(x, 0, z) = u(x, y, 0) = u(x, y, 1) = 0
    u(1, y, z) = y * sin(pi * z)
    u(x, 1, z) = x * sin(pi * z)
    (implemented as u(1, y, z) = u(x, 1, z) = x * y * sin(pi * z))
    (x,y,z) = [0,1]x[0,1]x[0,1]

    u(x, y, z) = x * y * sin(pi * z)
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        return ((torch.pi ** 2) * input[:, 0] * input[:, 1] * torch.sin(
            torch.pi * input[:, 2])).unsqueeze(1)

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        return (input[:, 0] * input[:, 1] * torch.sin(
            torch.pi * input[:, 2])).unsqueeze(1)

    # dirichlet bc

    @staticmethod
    def bc(model: models.NN) -> torch.Tensor:
        residual = model.output - Problem5.exact(model.input)
        return residual.pow(2).mean()


class Problem6:
    """
    Example 6 from
    https://www.researchgate.net/publication
    /266007367_An_Efficient_Direct_Method_to_Solve_the_Three_Dimensional_Poisson's_Equation
    -laplacian(u) = f
    -f(x, y, z) = -3 * pi^2 * sin(pi * x) * sin(pi * y) * sin(pi * z)
    u(0, x, z) = u(x, 0, z) = u(x, y, 0) = u(1, y, z) = u(x, 1, z) = u(x, y,
    1) = 0
    (x,y,z) = [0,1]x[0,1]x[0,1]

    u(x, y, z) = sin(pi * x) * sin(pi * y) * sin(pi * z)
    """

    @staticmethod
    def source(input: torch.Tensor) -> torch.Tensor:
        return (3 * (torch.pi ** 2) * torch.sin(
            torch.pi * input[:, 0]) * torch.sin(
            torch.pi * input[:, 1]) * torch.sin(
            torch.pi * input[:, 2])).unsqueeze(1)

    @staticmethod
    def exact(input: torch.Tensor) -> torch.Tensor:
        return (torch.sin(torch.pi * input[:, 0]) * torch.sin(
            torch.pi * input[:, 1]) * torch.sin(
            torch.pi * input[:, 2])).unsqueeze(1)

    # dirichlet bc

