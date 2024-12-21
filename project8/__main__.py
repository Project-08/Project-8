import argparse
import gc
import logging

import pyamgx

from project8.bench.sparse import bench_solve_tri
from project8.sparse import (AMGX_CONFIGS, AmgXSolve, CuPySolve,
                             CuSPARSETriSolve, TriSolve)


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(prog='project8')
    commands = parser.add_subparsers(dest="command")

    bench = commands.add_parser("bench")
    benchmark = bench.add_subparsers(dest="benchmark")
    solve_tri = benchmark.add_parser("solve-tri")

    solve_tri.add_argument("--min", type=int, default=10)
    solve_tri.add_argument("--max", type=int, default=1000)
    solve_tri.add_argument("--steps", type=int, default=10)
    solve_tri.add_argument("--attempts", type=int, default=10)
    solve_tri.add_argument(
        "--solvers",
        nargs='+',
        required=True,
    )
    solve_tri.add_argument("--plot", action='store_true', default=False)

    args = parser.parse_args()

    if args.command == "bench":
        if args.benchmark == "solve-tri":
            solvers: list[TriSolve] = []
            for solver in args.solvers:
                match solver:
                    case "cuSPARSE":
                        solvers.append(CuSPARSETriSolve())
                    case "CuPy":
                        solvers.append(CuPySolve())
                    case "AmgX":
                        solvers.extend(
                            (AmgXSolve(config) for config in AMGX_CONFIGS))
                    case _ if solver in AMGX_CONFIGS:
                        solvers.append(AmgXSolve(solver))
                    case _:
                        raise ValueError(f"Unrecognized solver: {solver}")

            bench_solve_tri(range(args.min, args.max,
                                  (args.max - args.min) // args.steps),
                            solvers=solvers,
                            attempts=args.attempts,
                            plot=args.plot)
            del solvers

    gc.collect()
    pyamgx.finalize()


if __name__ == "__main__":
    main()
