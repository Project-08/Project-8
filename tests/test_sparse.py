import os

import cupy as cp
import pytest

from project8.sparse import AmgXSolve, CuPySolve, CuSPARSETriSolve


@pytest.mark.skipif(os.getenv('NOGPU') is not None,
                    reason="Skipping GPU test.")
def test_cupy_tri_solve() -> None:
    d = 2 * cp.ones(3)
    dlu = -1 * cp.ones(2)
    b = cp.ones(3)
    solver = CuPySolve()
    x = solver.solve_tri(dlu, d, dlu, b)[0]
    assert cp.allclose(x, cp.array([1.5, 2., 1.5]))


@pytest.mark.skipif(os.getenv('NOGPU') is not None,
                    reason="Skipping GPU test.")
def test_cusparse_tri_solve() -> None:
    d = 2 * cp.ones(4)
    dlu = -1 * cp.ones(3)
    b = cp.ones(4)
    solver = CuSPARSETriSolve()
    x = solver.solve_tri(dlu, d, dlu, b)[0]
    assert cp.allclose(x, cp.array([2., 3., 3., 2.]))


@pytest.mark.skipif(os.getenv('NOGPU') is not None,
                    reason="Skipping GPU test.")
def test_amgx_tri_solve() -> None:
    d = 2 * cp.ones(4)
    dlu = -1 * cp.ones(3)
    b = cp.ones(4)
    solver = AmgXSolve()
    x = solver.solve_tri(dlu, d, dlu, b)[0]
    assert cp.allclose(x, cp.array([2., 3., 3., 2.]))
