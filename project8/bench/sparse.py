from collections import defaultdict
from logging import info
from typing import Sequence

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np

from project8.perf import Perf, average_performance
from project8.sparse import TriSolve


def bench_solve_tri(sizes: Sequence[int],
                    solvers: Sequence[TriSolve],
                    attempts: int = 1,
                    plot: bool = False) -> None:
    results: dict[str, list[list[Perf]]] = defaultdict(list)

    for n in sizes:
        d = 2 * cp.ones(n)
        dlu = -1 * cp.ones(n - 1)

        for solver in solvers:
            solver_name = str(solver)
            b = cp.random.rand(n)
            perfs = []
            for i in range(attempts + 1):
                perf = solver.solve_tri(dlu, d, dlu, b)[1]
                if i:  # discard the first measurement - let the GPU "warm up"
                    info(f"{solver_name} n={n} perf={perf}")
                    perfs.append(perf)
            results[solver_name].append(perfs)

    if plot:
        plt.figure()
        colormap = plt.cm.get_cmap('gist_rainbow', len(solvers))

        for (i, solver) in enumerate(solvers):
            solver_name = str(solver)
            solver_perfs = results[solver_name]
            color = colormap(i / len(solvers))
            avg_perfs = [average_performance(perfs) for perfs in solver_perfs]

            scatter_x = [n for n in sizes for _ in range(attempts)]

            plt.subplot(1, 3, 1)
            plt.scatter(
                scatter_x,
                [perf.total_ns for perfs in solver_perfs for perf in perfs],
                label=solver_name,
                marker='x',
                color=color)
            plt.plot(sizes, [perf.total_ns for perf in avg_perfs],
                     label=f'{solver_name} (mean)',
                     marker='o',
                     color=color)
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('Total Time (ns)')
            plt.title('Performance of Solvers (Total Time vs. Matrix Size)')
            plt.legend(fontsize="xx-small")
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.scatter(scatter_x,
                        np.array([
                            perf.gpu_ms for perfs in solver_perfs
                            for perf in perfs
                        ]),
                        label=solver_name,
                        marker='x',
                        color=color)
            plt.plot(sizes,
                     np.array([perf.gpu_ms for perf in avg_perfs]),
                     label=f'{solver_name} (mean)',
                     marker='o',
                     color=color)
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('GPU Time (ms)')
            plt.title('Performance of Solvers (GPU Time vs. Matrix Size)')
            plt.legend(fontsize="xx-small")
            plt.grid(True)

            plt.subplot(1, 3, 3)
            plt.scatter(scatter_x,
                        np.array([
                            perf.buffer_size for perfs in solver_perfs
                            for perf in perfs
                        ]),
                        label=solver_name,
                        marker='x',
                        color=color)
            plt.plot(sizes,
                     np.array([perf.buffer_size for perf in avg_perfs]),
                     label=f'{solver_name} (mean)',
                     marker='o',
                     color=color)
            plt.xlabel('Matrix Size (n)')
            plt.ylabel('Buffer Size (B)')
            plt.title('Performance of Solvers (Memory Usage vs. Matrix Size)')
            plt.legend(fontsize="xx-small")
            plt.grid(True)

        plt.show()
