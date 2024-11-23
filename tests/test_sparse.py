import cupy as cp

from project8.sparse import CuPySolve, CuSparseTriSolve


def test_cupy_tri_solve() -> None:
    d = 2 * cp.ones(3)
    dlu = -1 * cp.ones(2)
    b = cp.ones(3)
    solver = CuPySolve()
    x = solver.solve_tri(dlu, d, dlu, b)[0]
    assert cp.allclose(x, cp.array([1.5, 2., 1.5]))


def test_cusparse_tri_solve() -> None:
    d = 2 * cp.ones(4)
    dlu = -1 * cp.ones(3)
    b = cp.ones(4)
    solver = CuSparseTriSolve()
    x = solver.solve_tri(dlu, d, dlu, b)[0]
    assert cp.allclose(x, cp.array([2., 3., 3., 2.]))
