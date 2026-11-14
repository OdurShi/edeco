"""Command-line interface: ``edeco demo | solve | run | tables | figures``."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from edeco import __version__
from edeco.benchmarks import SUITES, get_problem
from edeco.benchmarks.cec2017 import DATA_ENV_VAR
from edeco.optimizer import ALGORITHMS, get_config, minimize

DATA_DIR_HELP = "directory with the CEC 2017 definition data (default: data/cec2017)"


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _function_id(text: str) -> str | int:
    return int(text) if text.lstrip("-").isdigit() else text


def build_parser() -> argparse.ArgumentParser:
    """Argument parser of the ``edeco`` command."""
    parser = argparse.ArgumentParser(
        prog="edeco",
        description="EDECO: an Enhanced Educational Competition Optimizer (re-implementation).",
    )
    parser.add_argument("--version", action="version", version=f"edeco {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="log progress")
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("demo", help="quick comparison on built-in test functions")
    demo.add_argument("--dim", type=_positive_int, default=10)
    demo.add_argument("--runs", type=_positive_int, default=5)
    demo.add_argument("--max-fes", type=_positive_int, default=20_000)
    demo.add_argument("--pop-size", type=_positive_int, default=100)
    demo.add_argument("--seed", type=int, default=0)

    solve = commands.add_parser("solve", help="one optimisation run on a benchmark problem")
    solve.add_argument("--suite", choices=SUITES, default="classical")
    solve.add_argument("--function", type=_function_id, default="shifted_sphere")
    solve.add_argument("--dim", type=_positive_int, default=10)
    solve.add_argument("--algorithm", choices=sorted(ALGORITHMS), default="EDECO")
    solve.add_argument("--pop-size", type=_positive_int, default=100)
    solve.add_argument("--max-fes", type=_positive_int, default=None, help="default: 5000 * D")
    solve.add_argument("--seed", type=int, default=0)
    solve.add_argument("--data-dir", type=Path, default=None, help=DATA_DIR_HELP)

    run = commands.add_parser("run", help="run an experiment described by a YAML file")
    run.add_argument("--data-dir", type=Path, default=None, help=DATA_DIR_HELP)
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--output", type=Path, default=None, help="default: results/<name>")
    run.add_argument("--runs", type=_positive_int, default=None, help="override the YAML")
    run.add_argument("--workers", type=_positive_int, default=None, help="override the YAML")
    run.add_argument("--seed", type=int, default=None, help="override the YAML")
    run.add_argument(
        "--functions", type=_function_id, nargs="+", default=None, help="override the YAML"
    )

    tables = commands.add_parser("tables", help="write result tables (Markdown and CSV)")
    tables.add_argument("--results", type=Path, required=True)
    tables.add_argument("--reference", default=None, help="algorithm compared by the rank-sum")

    figures = commands.add_parser("figures", help="write convergence, box and ranking plots")
    figures.add_argument("--results", type=Path, required=True)
    figures.add_argument("--dpi", type=_positive_int, default=150)
    return parser


def _demo(args: argparse.Namespace) -> int:
    functions = ["shifted_sphere", "rosenbrock", "shifted_rastrigin", "shifted_ackley"]
    names = list(ALGORITHMS)
    print(
        f"Mean best value over {args.runs} runs, D = {args.dim}, N = {args.pop_size}, "
        f"{args.max_fes} evaluations per run (optimum 0 for every function)"
    )
    print(f"{'function':<20}" + "".join(f"{name:>14}" for name in names))
    start = time.perf_counter()
    for function in functions:
        problem = get_problem("classical", function, args.dim)
        means = []
        for name in names:
            finals = [
                minimize(
                    problem.objective,
                    problem.bounds,
                    problem.dim,
                    pop_size=args.pop_size,
                    max_fes=args.max_fes,
                    seed=args.seed + run,
                    config=get_config(name),
                ).best_f
                for run in range(args.runs)
            ]
            means.append(float(np.mean(finals)))
        print(f"{function:<20}" + "".join(f"{mean:>14.4e}" for mean in means))
    print(f"finished in {time.perf_counter() - start:.1f} s")
    return 0


def _solve(args: argparse.Namespace) -> int:
    problem = get_problem(args.suite, args.function, args.dim)
    max_fes = args.max_fes if args.max_fes is not None else 5000 * problem.dim
    result = minimize(
        problem.objective,
        problem.bounds,
        problem.dim,
        pop_size=args.pop_size,
        max_fes=max_fes,
        seed=args.seed,
        config=get_config(args.algorithm),
    )
    print(f"problem      : {problem.suite} / {problem.name} (D = {problem.dim})")
    print(f"algorithm    : {args.algorithm} (N = {args.pop_size}, seed = {args.seed})")
    print(f"evaluations  : {result.n_evals} in {result.n_iterations} iterations")
    print(f"best value   : {result.best_f:.10g}")
    if problem.f_opt is not None:
        print(f"error f - f* : {result.best_f - problem.f_opt:.6e}")
    if args.suite == "engineering":
        from edeco.benchmarks.engineering import get_engineering_problem

        design = get_engineering_problem(problem.name)
        print(f"objective    : {design.objective(result.best_x):.10g}")
        print(f"max violation: {design.max_violation(result.best_x):.3e}")
        solution = design.repair(result.best_x)
    else:
        solution = result.best_x
    print("best x       : " + np.array2string(solution, precision=6, max_line_width=100))
    return 0


def _run(args: argparse.Namespace) -> int:
    from edeco.experiments.config import load_config
    from edeco.experiments.runner import run_experiment

    config = load_config(
        args.config, runs=args.runs, workers=args.workers, seed=args.seed, functions=args.functions
    )
    output = args.output if args.output is not None else Path("results") / config.name
    run_experiment(config, output)
    print(f"raw results written to {output}")
    return 0


def _tables(args: argparse.Namespace) -> int:
    from edeco.analysis.tables import write_all_tables

    for path in write_all_tables(args.results, args.reference):
        print(path)
    return 0


def _figures(args: argparse.Namespace) -> int:
    from edeco.analysis.plots import write_all_figures

    for path in write_all_figures(args.results, args.dpi):
        print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``edeco`` console script."""
    args = build_parser().parse_args(argv)
    if getattr(args, "data_dir", None) is not None:
        # The environment variable also reaches the worker processes of `edeco run`.
        os.environ[DATA_ENV_VAR] = str(args.data_dir)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    handlers = {
        "demo": _demo,
        "solve": _solve,
        "run": _run,
        "tables": _tables,
        "figures": _figures,
    }
    return handlers[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
