from logging import info
from typing import Sequence

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from ..perf import Perf, average_performance
from ..sparse import TriSolve


def bench_solve_tri(sizes: Sequence[int],
                    solvers: Sequence[TriSolve],
                    attempts: int = 1,
                    plot: bool = False) -> None:

    results: dict[tuple[int, str], list[Perf]] = {}

    for n in sizes:
        d = 2 * cp.ones(n)
        dlu = -1 * cp.ones(n - 1)

        for solver in solvers:
            b = cp.random.rand(n)
            perfs = []
            for i in range(attempts + 1):
                perf = solver.solve_tri(dlu, d, dlu, b)[1]
                if i:  # discard the first measurement - let the GPU "warm up"
                    info(f"{type(solver).__name__}: n={n}, perf={perf}")
                    perfs.append(perf)
            results[(n, type(solver).__name__)] = perfs

    if plot:
        plt.figure(figsize=(12, 6))

        for solver in solvers:
            solver_name = type(solver).__name__
            solver_perfs = [results[(n, solver_name)] for n in sizes]

            mean_ns_perfs = [
                average_performance(perfs).total_ns for perfs in solver_perfs
            ]

            mean_gpu_perfs = [
                average_performance(perfs).gpu_ms for perfs in solver_perfs
            ]

            all_ns_perfs = [
                perf.total_ns for perfs in solver_perfs for perf in perfs
            ]

            all_gpu_perfs = [
                perf.gpu_ms for perfs in solver_perfs for perf in perfs
            ]

            scatter_x = [n for n in sizes for _ in range(attempts)]

            plt.subplot(1, 2, 1)
            plt.scatter(scatter_x,
                        all_ns_perfs,
                        label=f'{solver_name} (scatter)',
                        s=20)
            plt.plot(sizes,
                     mean_ns_perfs,
                     label=f'{solver_name} (mean)',
                     marker='o',
                     linestyle='-',
                     linewidth=2)
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('Total Time (ns)')
            plt.title('Performance of Solvers (Total Time vs. Matrix Size)')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.scatter(scatter_x,
                        np.array(all_gpu_perfs),
                        label=f'{solver_name} (scatter)',
                        s=20)
            plt.plot(sizes,
                     np.array(mean_gpu_perfs),
                     label=f'{solver_name} (mean)',
                     marker='o',
                     linestyle='-',
                     linewidth=2)
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('GPU Time (ms)')
            plt.title('Performance of Solvers (GPU Time vs. Matrix Size)')
            plt.legend()
            plt.grid(True)

        plt.tight_layout()
        plt.show()
