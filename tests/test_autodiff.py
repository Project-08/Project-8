from project8.neural_network import utils
from project8.neural_network import autograd_wrapper as aw
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


def setup() -> tuple[aw.Differentiator, torch.Tensor, torch.Tensor]:
    diff = aw.Differentiator(f)
    coord_space = utils.ParameterSpace([[-1, 1], [-1, 1]], 'cpu')
    grid = coord_space.fgrid(10).requires_grad_(True)
    diff.set_input(grid)
    x = diff.input()[:, 0].unsqueeze(1)
    y = diff.input()[:, 1].unsqueeze(1)
    return diff, x, y


def test_dfdy() -> None:
    diff, x, y = setup()
    dy = diff(1)
    dy_true = dfdy(x, y)
    assert torch.allclose(dy, dy_true, atol=1e-5)


def test_d2fdxx() -> None:
    diff, x, y = setup()
    d2x = diff(0, 0)
    d2x_true = d2fdxx(x, y)
    assert torch.allclose(d2x, d2x_true, atol=1e-5)


def test_d3fdxxx() -> None:
    diff, x, y = setup()
    d3x = diff(0, 0, 0)
    d3x_true = d3fdxxx(x, y)
    assert torch.allclose(d3x, d3x_true, atol=1e-5)


def test_d2fdyy() -> None:
    diff, x, y = setup()
    d2y = diff(1, 1)
    d2y_true = d2fdyy(x, y)
    assert torch.allclose(d2y, d2y_true, atol=1e-5)


def test_d2fdxy() -> None:
    diff, x, y = setup()
    d2xy = diff(0, 1)
    d2xy_true = d2fdxy(x, y)
    assert torch.allclose(d2xy, d2xy_true, atol=1e-5)


def test_d2fdyx() -> None:
    diff, x, y = setup()
    d2yx = diff(1, 0)
    d2yx_true = d2fdyx(x, y)
    assert torch.allclose(d2yx, d2yx_true, atol=1e-5)


def test_laplacian() -> None:
    diff, x, y = setup()
    lap = diff.laplacian()
    lap_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(lap, lap_true, atol=1e-5)


def test_grad_div() -> None:
    diff, x, y = setup()
    lap = diff.divergence(diff.gradient())
    lap_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(lap, lap_true, atol=1e-5)
