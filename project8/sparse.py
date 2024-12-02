import os
import re
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

AMGX_CONFIG_PATH = os.environ.get("AMGX_CONFIG_PATH",
                                  "/usr/local/lib/configs/")

# Extremely slow configs have been commented out
AMGX_CONFIGS = (
    "AGGREGATION_DILU",
    "AGGREGATION_GS",
    "AGGREGATION_JACOBI",
    "AGGREGATION_MULTI_PAIRWISE",
    # "CG_DILU",
    "CLASSICAL_CG_CYCLE",
    "CLASSICAL_CGF_CYCLE",
    "CLASSICAL_F_CYCLE",
    "CLASSICAL_V_CYCLE",
    "CLASSICAL_W_CYCLE",
    "FGMRES_AGGREGATION_DILU",
    "FGMRES_AGGREGATION_JACOBI",
    "FGMRES_CLASSICAL_AGGRESSIVE_HMIS",
    "FGMRES_CLASSICAL_AGGRESSIVE_PMIS",
    "FGMRES_NOPREC",
    "GMRES_AMG_D2",
    "IDR_DILU",
    "IDRMSYNC_DILU",
    # "PBICGSTAB_AGGREGATION_W_JACOBI",
    "PBICGSTAB_CLASSICAL_JACOBI",
    "PBICGSTAB_NOPREC",
    "PCG_AGGREGATION_JACOBI",
    "PCG_CLASSICAL_F_JACOBI",
    "PCG_CLASSICAL_V_JACOBI",
    "PCG_CLASSICAL_W_JACOBI",
    # "PCG_DILU",
    "PCGF_CLASSICAL_F_JACOBI",
    "PCGF_CLASSICAL_V_JACOBI",
    "PCGF_CLASSICAL_W_JACOBI",
    "PCG_NOPREC",
)

_AMGX_MEMORY_PAT = re.compile(r"Maximum Memory Usage:\s*(?P<value>[\d\.]+) GB")
_AMGX_TIME_PAT = re.compile(r"Total Time: (?P<total_time>[\d.]+)\s*\n.*\n"
                            r"\s*solve: (?P<solve_time>[\d.]+)")


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

    def __str__(self) -> str:
        return "cuSPARSE dgtsv2_nopivot"


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

    def __str__(self) -> str:
        return "CuPy spsolve"


class AmgXSolve(Solve, TriSolve):
    """Solver of Ax=b systems using AmgX."""

    def __init__(self, config_name: str) -> None:
        self._config_name = config_name
        self._cfg = pyamgx.Config().create_from_file(
            os.path.join(AMGX_CONFIG_PATH, config_name + ".json"))
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
            warn("AmgXSolve.solve: converting matrix to csr")
            a = a.tocsr()

        perf = Perf(0, buffer_size_is_tight=True)

        def print_callback(msg: str) -> None:
            mem = re.search(_AMGX_MEMORY_PAT, msg)
            if mem is not None:
                perf.buffer_size = int(
                    float(mem.group("value")) * 1_000_000_000)
                return
            time_ = re.search(_AMGX_TIME_PAT, msg)
            if time_ is not None:
                perf.total_ns = int(
                    float(time_.group("total_time")) * 1_000_000_000)
                perf.gpu_ms = float(time_.group("solve_time")) * 1_000

        pyamgx.register_print_callback(print_callback)

        A = pyamgx.Matrix().create(self._rsc)
        B = pyamgx.Vector().create(self._rsc)
        X = pyamgx.Vector().create(self._rsc)

        x = np.zeros(b.shape[0])

        A.upload_CSR(a)
        B.upload_raw(b.data.ptr, b.shape[0])
        X.upload(x)

        self._solver.setup(A)
        self._solver.solve(B, X, zero_initial_guess=True)

        X.download(x)

        A.destroy()
        X.destroy()
        B.destroy()

        return (x, perf)

    def __str__(self) -> str:
        return f"AmgX ({self._config_name})"
