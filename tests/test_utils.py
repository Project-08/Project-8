from project8.neural_network import utils
import torch
from typing import List


def count_points_on_planes(points: torch.Tensor,
                           tolerance: float = 1e-6) -> List[int | float]:
    counts = []
    for i in range(points.shape[1]):
        counts.append(torch.sum(
            torch.abs(points[:, points.shape[1] - i - 1]) <= tolerance).item())
    return counts


def test_parameter_space() -> None:
    device = utils.get_device('cpu')
    domain = [[0, 1], [0, 1], [0, 1]]
    n = 10000
    space = utils.ParameterSpace(domain, device)
    bound_sel = torch.tensor([[1, 1], [1, 1], [0, 0]], device=device)
    tmr = utils.Timer()
    tmr.start()
    loc = space.select_bndry_rand(n, bound_sel)
    tmr.stop()
    tmr.read()
    assert (count_points_on_planes(loc)[0] == 0
            and count_points_on_planes(loc)[1] > 0
            and count_points_on_planes(loc)[2] > 0
            and tmr.t_elapsed < 0.3)
