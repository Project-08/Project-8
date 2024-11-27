from project8.neural_network import models
from project8.neural_network import utils
import torch

def f(x, y):
    return x**3 + 3*y**2 + 2*x*y + 2*y**3 + 3*x + 2

def dfdx(x, y):
    return 3*x**2 + 2*y + 3

def dfdy(x, y):
    return 6*y + 2*x + 6*y**2

def d2fdxx(x, y):
    return 6*x

def d3fdxxx(x, y):
    return 6 + x*0

def d2fdyy(x, y):
    return 6 + 12*y

def d2fdxy(x, y):
    return x*0 + 2

def d2fdyx(x, y):
    return x*0 + 2

def def_model() -> models.diff_NN:
    model = models.diff_NN.empty()
    coord_space = utils.ParameterSpace([[-1, 1], [-1, 1]], 'cpu')
    grid = coord_space.fgrid(10).requires_grad_(True)
    model.input = grid
    x = grid[:, 0].unsqueeze(1)
    y = grid[:, 1].unsqueeze(1)
    model.output = f(x, y)
    return model

def test_dfdx() -> None:
    model = def_model()
    dx = model.diff(0)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    dx_true = dfdx(x, y)
    assert torch.allclose(dx, dx_true, atol=1e-5)

def test_dfdy() -> None:
    model = def_model()
    dy = model.diff(1)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    dy_true = dfdy(x, y)
    assert torch.allclose(dy, dy_true, atol=1e-5)

def test_d2fdxx() -> None:
    model = def_model()
    d2x = model.diff(0, 0)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    d2x_true = d2fdxx(x, y)
    assert torch.allclose(d2x, d2x_true, atol=1e-5)

def test_d3fdxxx() -> None:
    model = def_model()
    d3x = model.diff(0, 0, 0)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    d3x_true = d3fdxxx(x, y)
    assert torch.allclose(d3x, d3x_true, atol=1e-5)

def test_d2fdyy() -> None:
    model = def_model()
    d2y = model.diff(1, 1)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    d2y_true = d2fdyy(x, y)
    assert torch.allclose(d2y, d2y_true, atol=1e-5)

def test_d2fdxy() -> None:
    model = def_model()
    d2xy = model.diff(0, 1)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    d2xy_true = d2fdxy(x, y)
    assert torch.allclose(d2xy, d2xy_true, atol=1e-5)

def test_d2fdyx() -> None:
    model = def_model()
    d2yx = model.diff(1, 0)
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    d2yx_true = d2fdyx(x, y)
    assert torch.allclose(d2yx, d2yx_true, atol=1e-5)

def test_laplacian() -> None:
    model = def_model()
    lap = model.laplacian()
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    lap_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(lap, lap_true, atol=1e-5)

def test_grad_div() -> None:
    model = def_model()
    lap = model.divergence(model.gradient())
    x = model.input[:, 0].unsqueeze(1)
    y = model.input[:, 1].unsqueeze(1)
    lap_true = d2fdxx(x, y) + d2fdyy(x, y)
    assert torch.allclose(lap, lap_true, atol=1e-5)