from abc import ABC, abstractmethod
from logging import warn
from time import process_time_ns
from typing import cast

import cupy as cp
from cupy.cuda import Event, cusparse
from cupyx.scipy.sparse import csr_matrix, diags, spmatrix
from cupyx.scipy.sparse.linalg import spsolve

from .perf import Perf


def spmatrix_size(matrix: spmatrix) -> int:
    """Calculate the memory size in bytes of a CuPy sparse matrix."""
    if isinstance(matrix, csr_matrix):
        return cast(
            int,
            matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes)
    raise TypeError(f"Unsupported sparse matrix type: {type(matrix)}")


class Solve(ABC):
    """Solver of Ax=b systems."""

    @abstractmethod
    def solve(self, a: spmatrix, b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        pass


class TriSolve(ABC):
    """Solver of Ax=b systems for a tridiagonal A."""

    @abstractmethod
    def solve_tri(self, dl: cp.ndarray, d: cp.ndarray, du: cp.ndarray,
                  b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        pass


class CuSparseTriSolve(TriSolve):
    """Solver of Ax=b systems for a tridiagonal A using cuSPARSE."""

    def __init__(self) -> None:
        self._handle = cusparse.create()

    def solve_tri(self, dl: cp.ndarray, d: cp.ndarray, du: cp.ndarray,
                  b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        n = d.shape[0]
        start_event = Event()
        end_event = Event()
        start_ns = process_time_ns()
        start_event.record()
        buffer_size = cusparse.dgtsv2_nopivot_bufferSizeExt(
            self._handle, n, 1, dl.data.ptr, d.data.ptr, du.data.ptr,
            b.data.ptr, n)
        p_buffer = cp.empty(buffer_size, dtype=cp.uint8)
        start_event.record()
        cusparse.dgtsv2_nopivot(self._handle, n, 1, dl.data.ptr, d.data.ptr,
                                du.data.ptr, b.data.ptr, n, p_buffer.data.ptr)
        end_event.record()
        end_event.synchronize()
        x = b.copy()
        end_ns = process_time_ns()
        return (x,
                Perf.from_measurements((start_ns, end_ns),
                                       (start_event, end_event),
                                       buffer_size,
                                       buffer_size_is_tight=True))


class CuPySolve(Solve, TriSolve):
    """Solver of Ax=b systems using CuPy."""

    def solve(self, a: spmatrix, b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        if not isinstance(a, csr_matrix):
            warn("CuPySolve.solve: converting matrix to csr")
            a = a.tocsr()
        start_ns = process_time_ns()
        x = spsolve(a, b)
        end_ns = process_time_ns()
        return (x,
                Perf.from_measurements((start_ns, end_ns), None,
                                       spmatrix_size(a)))

    def solve_tri(self, dl: cp.ndarray, d: cp.ndarray, du: cp.ndarray,
                  b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        start_ns = process_time_ns()
        a = diags([dl, d, du], [-1, 0, 1], format="csr")
        x = spsolve(a, b)
        end_ns = process_time_ns()
        return (x,
                Perf.from_measurements((start_ns, end_ns), None,
                                       spmatrix_size(a)))
