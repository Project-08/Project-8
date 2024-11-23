from collections import defaultdict
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
    results: dict[str, list[list[Perf]]] = defaultdict(list)

    for n in sizes:
        d = 2 * cp.ones(n)
        dlu = -1 * cp.ones(n - 1)

        for solver in solvers:
            solver_name = type(solver).__name__
            b = cp.random.rand(n)
            perfs = []
            for i in range(attempts + 1):
                perf = solver.solve_tri(dlu, d, dlu, b)[1]
                if i:  # discard the first measurement - let the GPU "warm up"
                    info(f"{solver_name} n={n} perf={perf}")
                    perfs.append(perf)
            results[solver_name].append(perfs)

    if plot:
        plt.figure(figsize=(12, 6))

        for solver in solvers:
            solver_name = type(solver).__name__
            solver_perfs = results[solver_name]
            avg_perfs = [average_performance(perfs) for perfs in solver_perfs]

            scatter_x = [n for n in sizes for _ in range(attempts)]

            plt.subplot(1, 2, 1)
            plt.scatter(
                scatter_x,
                [perf.total_ns for perfs in solver_perfs for perf in perfs],
                label=solver_name,
                marker='x')
            plt.plot(sizes, [perf.total_ns for perf in avg_perfs],
                     label=f'{solver_name} (mean)',
                     marker='o')
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('Total Time (ns)')
            plt.title('Performance of Solvers (Total Time vs. Matrix Size)')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 2, 2)
            plt.scatter(scatter_x,
                        np.array([
                            perf.gpu_ms for perfs in solver_perfs
                            for perf in perfs
                        ]),
                        label=solver_name,
                        marker='x')
            plt.plot(sizes,
                     np.array([perf.gpu_ms for perf in avg_perfs]),
                     label=f'{solver_name} (mean)',
                     marker='o')
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('GPU Time (ms)')
            plt.title('Performance of Solvers (GPU Time vs. Matrix Size)')
            plt.legend()
            plt.grid(True)

        plt.show()
