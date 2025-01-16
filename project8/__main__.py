import argparse
import logging


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
            from project8.bench.sparse import bench_solve_tri
            from project8.sparse import CuPySolve, CuSparseTriSolve, TriSolve
            solvers: list[TriSolve] = []
            for solver in args.solvers:
                match solver:
                    case "cuSPARSE":
                        solvers.append(CuSparseTriSolve())
                    case "CuPy":
                        solvers.append(CuPySolve())

            bench_solve_tri(range(args.min, args.max,
                                  (args.max - args.min) // args.steps),
                            solvers=solvers,
                            attempts=args.attempts,
                            plot=args.plot)


if __name__ == "__main__":
    main()
