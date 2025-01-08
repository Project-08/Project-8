from project8.neural_network import utils
from project8.neural_network import differential_operators as diffop
import torch


def f(coords: torch.Tensor) -> torch.Tensor:
    x = coords[:, 0].unsqueeze(1)
    y = coords[:, 1].unsqueeze(1)
    return x.pow(3) + 3 * y.pow(2) + 2 * x * y + 2 * y.pow(3) + 3 * x + 2


def dfdx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 3 * x.pow(2) + 2 * y + 3


def dfdy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 6 * y + 2 * x + 6 * y.pow(2)


def d2fdxx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 6 * x


def d3fdxxx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 6 + x * 0


def d2fdyy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return 6 + 12 * y


def d2fdxy(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * 0 + 2


def d2fdyx(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return x * 0 + 2


def setup() -> tuple[diffop.AbstractIO, torch.Tensor, torch.Tensor]:
    io = diffop.AbstractIO()
    coord_space = utils.ParameterSpace([[-1, 1], [-1, 1]], 'cpu')
    io.input = coord_space.fgrid(10).requires_grad_(True)
    io.output = f(io.input)
    x = io.input[:, 0].unsqueeze(1)
    y = io.input[:, 1].unsqueeze(1)
    return io, x, y


def test_dfdx() -> None:
    io, x, y = setup()
    dx = diffop.partial_derivative(io, 0)
    dx_true = dfdx(x, y)
    assert torch.allclose(dx, dx_true, atol=1e-5)


def test_dfdy() -> None:
    io, x, y = setup()
    dy = diffop.partial_derivative(io, 1)
    dy_true = dfdy(x, y)
    assert torch.allclose(dy, dy_true, atol=1e-5)


def test_d2fdxx() -> None:
    io, x, y = setup()
    d2x = diffop.partial_derivative(io, 0, 0)
    d2x_true = d2fdxx(x, y)
    assert torch.allclose(d2x, d2x_true, atol=1e-5)


def test_d3fdxxx() -> None:
    io, x, y = setup()
    d3x = diffop.partial_derivative(io, 0, 0, 0)
    d3x_true = d3fdxxx(x, y)
    assert torch.allclose(d3x, d3x_true, atol=1e-5)


def test_d2fdyy() -> None:
    io, x, y = setup()
    d2y = diffop.partial_derivative(io, 1, 1)
    d2y_true = d2fdyy(x, y)
    assert torch.allclose(d2y, d2y_true, atol=1e-5)


def test_d2fdxy() -> None:
    io, x, y = setup()
    d2xy = diffop.partial_derivative(io, 0, 1)
    d2xy_true = d2fdxy(x, y)
    assert torch.allclose(d2xy, d2xy_true, atol=1e-5)


def test_d2fdyx() -> None:
    io, x, y = setup()
    d2yx = diffop.partial_derivative(io, 1, 0)
    d2yx_true = d2fdyx(x, y)
    assert torch.allclose(d2yx, d2yx_true, atol=1e-5)


def test_laplacian() -> None:
    io, x, y = setup()
    laplace = diffop.laplacian(io)
    laplace_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(laplace, laplace_true, atol=1e-5)


def test_grad_div() -> None:
    io, x, y = setup()
    grad = diffop.gradient(io)
    div = diffop.divergence(io, grad)
    div_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(div, div_true, atol=1e-5)
