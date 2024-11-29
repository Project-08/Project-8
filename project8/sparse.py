import weakref
from abc import ABC, abstractmethod
from logging import warn
from time import process_time_ns
from typing import cast

import cupy as cp
import numpy as np
import pyamgx
from cupy.cuda import Event, cusparse
from cupyx.scipy.sparse import diags, isspmatrix_csr, spmatrix
from cupyx.scipy.sparse.linalg import spsolve

from .perf import Perf


def spmatrix_size(matrix: spmatrix) -> int:
    """Calculate the memory size in bytes of a CuPy sparse matrix."""
    if isspmatrix_csr(matrix):
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


class CuSPARSETriSolve(TriSolve):
    """Solver of Ax=b systems for a tridiagonal A using cuSPARSE."""

    def __init__(self) -> None:
        self._handle = cusparse.create()

    def solve_tri(self, dl: cp.ndarray, d: cp.ndarray, du: cp.ndarray,
                  b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        n = d.shape[0]
        dl = cp.concatenate((cp.zeros(1), dl), dtype=cp.float64)
        d = d.astype(cp.float64)
        du = cp.concatenate((du, cp.zeros(1)), dtype=cp.float64)
        b = b.astype(cp.float64)

        start_event = Event()
        end_event = Event()
        start_ns = process_time_ns()
        buffer_size = cusparse.dgtsv2_nopivot_bufferSizeExt(
            self._handle, n, 1, dl.data.ptr, d.data.ptr, du.data.ptr,
            b.data.ptr, n)
        p_buffer = cp.zeros(buffer_size, dtype=cp.uint8)
        start_event.record()
        cusparse.dgtsv2_nopivot(self._handle, n, 1, dl.data.ptr, d.data.ptr,
                                du.data.ptr, b.data.ptr, n, p_buffer.data.ptr)
        end_event.record()
        end_event.synchronize()
        end_ns = process_time_ns()
        return (b,
                Perf.from_measurements((start_ns, end_ns),
                                       (start_event, end_event),
                                       buffer_size,
                                       buffer_size_is_tight=True))


class CuPySolve(Solve, TriSolve):
    """Solver of Ax=b systems using CuPy."""

    def solve(self, a: spmatrix, b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        if not isspmatrix_csr(a):
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


class AmgXSolve(Solve, TriSolve):
    """Solver of Ax=b systems using AmgX."""

    def __init__(self) -> None:
        self._cfg = pyamgx.Config().create_from_dict({
            "config_version": 2,
            "determinism_flag": 1,
            "exception_handling": 1,
            "solver": {
                "monitor_residual": 1,
                "solver": "BICGSTAB",
                "convergence": "RELATIVE_INI_CORE",
                "preconditioner": {
                    "solver": "NOSOLVER"
                }
            }
        })
        self._rsc = pyamgx.Resources().create_simple(self._cfg)
        self._solver = pyamgx.Solver().create(self._rsc, self._cfg)
        self._finalizer = weakref.finalize(self, self._destructor)

    def _destructor(self) -> None:
        self._solver.destroy()
        self._rsc.destroy()
        self._cfg.destroy()

    def solve_tri(self, dl: cp.ndarray, d: cp.ndarray, du: cp.ndarray,
                  b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        start_ns = process_time_ns()
        a = diags([dl, d, du], [-1, 0, 1], format="csr")
        (x, perf) = self.solve(a, b)
        end_ns = process_time_ns()
        perf.total_ns = end_ns - start_ns
        return (x, perf)

    def solve(self, a: spmatrix, b: cp.ndarray) -> tuple[cp.ndarray, Perf]:
        if not isspmatrix_csr(a):
            warn("CuPySolve.solve: converting matrix to csr")
            a = a.tocsr()

        start_event = Event()
        end_event = Event()
        start_ns = process_time_ns()

        A = pyamgx.Matrix().create(self._rsc)
        B = pyamgx.Vector().create(self._rsc)
        X = pyamgx.Vector().create(self._rsc)

        x = np.zeros(b.shape[0])

        A.upload_CSR(a)
        B.upload_raw(b.data.ptr, b.shape[0])
        X.upload(x)

        start_event.record()
        self._solver.setup(A)
        self._solver.solve(B, X, zero_initial_guess=True)
        end_event.record()
        end_event.synchronize()

        X.download(x)

        A.destroy()
        X.destroy()
        B.destroy()

        end_ns = process_time_ns()
        return (x,
                Perf.from_measurements((start_ns, end_ns),
                                       (start_event, end_event), None))
