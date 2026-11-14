"""Experiment runner: independent runs of several algorithms on a benchmark suite.

Raw results are written to the output directory as

* ``runs.csv``    one row per run with the scalar outcomes,
* ``runs.jsonl``  one JSON record per run including the best solution and the
  convergence history sampled at ``history_points`` evaluation counts,
* ``config.json`` the resolved protocol and the versions of the numerical libraries.

The tables and figures of ``edeco.analysis`` are computed from these files only.
"""

from __future__ import annotations

import csv
import json
import logging
import multiprocessing
import platform
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import scipy

import edeco
from edeco.benchmarks import engineering, get_problem
from edeco.experiments.config import ExperimentConfig
from edeco.optimizer import ECOConfig, minimize

logger = logging.getLogger(__name__)

CSV_FIELDS = [
    "suite",
    "function",
    "dim",
    "algorithm",
    "run",
    "seed",
    "best_f",
    "error",
    "n_evals",
    "n_iterations",
    "pop_size",
    "time_s",
    "eda_samples",
    "eda_survivors",
    "objective",
    "max_violation",
    "feasible",
]


def checkpoint_evaluations(max_fes: int, n_points: int) -> np.ndarray:
    """Evaluation counts (1-based) at which the convergence history is stored."""
    return np.unique(np.linspace(1, max_fes, n_points).round().astype(int))


def sample_history(history: np.ndarray, checkpoints: np.ndarray) -> list[float]:
    """Best-so-far values at the given evaluation counts (ten significant digits)."""
    index = np.minimum(checkpoints, history.size) - 1
    return [float(f"{v:.10g}") for v in history[index]]


def run_single(task: dict[str, Any]) -> dict[str, Any]:
    """Execute one optimisation run described by ``task`` (worker entry point)."""
    options = {"penalty": task["penalty"]} if task["penalty"] is not None else {}
    problem = get_problem(task["suite"], task["function"], task["dim"], **options)
    config = ECOConfig(**task["algorithm_config"])
    start = time.perf_counter()
    result = minimize(
        problem.objective,
        problem.bounds,
        problem.dim,
        pop_size=task["pop_size"],
        max_fes=task["max_fes"],
        seed=task["seed"],
        config=config,
    )
    elapsed = time.perf_counter() - start
    record: dict[str, Any] = {
        "suite": task["suite"],
        "function": problem.name,
        "dim": problem.dim,
        "algorithm": task["algorithm"],
        "run": task["run"],
        "seed": task["seed"],
        "best_f": result.best_f,
        "error": result.best_f - problem.f_opt if problem.f_opt is not None else None,
        "n_evals": result.n_evals,
        "n_iterations": result.n_iterations,
        "pop_size": task["pop_size"],
        "time_s": round(elapsed, 4),
        "eda_samples": int(result.info["eda_samples"]),
        "eda_survivors": int(result.info["eda_survivors"]),
        "objective": None,
        "max_violation": None,
        "feasible": None,
        "best_x": [float(v) for v in result.best_x],
        "history_evals": [int(v) for v in task["checkpoints"]],
        "history": sample_history(result.history, np.asarray(task["checkpoints"])),
    }
    if task["suite"] == "engineering":
        design = engineering.get_engineering_problem(problem.name)
        record["objective"] = design.objective(result.best_x)
        record["max_violation"] = design.max_violation(result.best_x)
        record["feasible"] = design.is_feasible(result.best_x)
        record["best_x"] = [float(v) for v in design.repair(result.best_x)]
    return record


def build_tasks(config: ExperimentConfig) -> list[dict[str, Any]]:
    """Expand a configuration into the list of independent runs."""
    algorithms = config.algorithm_configs()
    population_sizes = config.population_sizes()
    tasks = []
    for function_index, function in enumerate(config.function_list()):
        options = {"penalty": config.penalty} if config.penalty is not None else {}
        dim = get_problem(config.suite, function, config.dim, **options).dim
        max_fes = config.budget(dim)
        checkpoints = checkpoint_evaluations(max_fes, config.history_points).tolist()
        for run in range(config.runs):
            # One seed per (function, run), shared by all algorithms.
            seed = int(
                np.random.SeedSequence([config.seed, function_index, run]).generate_state(1)[0]
            )
            for label, algorithm_config in algorithms.items():
                tasks.append(
                    {
                        "suite": config.suite,
                        "function": function,
                        "dim": config.dim,
                        "algorithm": label,
                        "algorithm_config": asdict(algorithm_config),
                        "pop_size": population_sizes[label],
                        "max_fes": max_fes,
                        "run": run,
                        "seed": seed,
                        "penalty": config.penalty,
                        "checkpoints": checkpoints,
                    }
                )
    return tasks


def _sort_key(record: dict[str, Any], function_order: dict[str, int], algorithms: list[str]):
    return (
        function_order[record["function"]],
        algorithms.index(record["algorithm"]),
        record["run"],
    )


def run_experiment(config: ExperimentConfig, output_dir: str | Path) -> Path:
    """Run every (function, algorithm, run) combination and write the raw results.

    Returns the output directory.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(config)
    logger.info("experiment %s: %d runs on %d worker(s)", config.name, len(tasks), config.workers)

    start = time.perf_counter()
    records: list[dict[str, Any]] = []
    if config.workers == 1:
        for done, task in enumerate(tasks, start=1):
            records.append(run_single(task))
            if done % 50 == 0:
                logger.info("%d / %d runs finished", done, len(tasks))
    else:
        context = multiprocessing.get_context("spawn")
        with context.Pool(config.workers) as pool:
            for done, record in enumerate(pool.imap_unordered(run_single, tasks), start=1):
                records.append(record)
                if done % 50 == 0:
                    logger.info("%d / %d runs finished", done, len(tasks))
    wall_time = time.perf_counter() - start

    algorithms = list(config.algorithm_configs())
    names = [get_problem_name(config, f) for f in config.function_list()]
    function_order = {name: i for i, name in enumerate(names)}
    records.sort(key=lambda r: _sort_key(r, function_order, algorithms))

    with open(output / "runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if record[k] is None else record[k]) for k in CSV_FIELDS})
    with open(output / "runs.jsonl", "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    metadata = {
        "config": config.to_dict(),
        "algorithms": {k: asdict(v) for k, v in config.algorithm_configs().items()},
        "population_sizes": config.population_sizes(),
        "n_runs_total": len(records),
        "wall_time_s": round(wall_time, 1),
        "versions": {
            "edeco": edeco.__version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output / "config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    logger.info("wrote %d runs to %s in %.1f s", len(records), output, wall_time)
    return output


def get_problem_name(config: ExperimentConfig, function: str | int) -> str:
    """Display name of a function identifier (``5`` -> ``"F5"`` for CEC 2017)."""
    return f"F{int(function)}" if config.suite == "cec2017" else str(function)
