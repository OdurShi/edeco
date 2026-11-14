"""End-to-end tests: experiment runner, tables, figures and the command-line interface."""

from __future__ import annotations

import csv
import json

import pytest

from edeco.analysis import stats
from edeco.analysis.plots import write_all_figures
from edeco.analysis.tables import write_all_tables
from edeco.cli import main
from edeco.experiments.config import ExperimentConfig, load_config
from edeco.experiments.runner import (
    build_tasks,
    checkpoint_evaluations,
    run_experiment,
    sample_history,
)


def tiny_config(**overrides) -> ExperimentConfig:
    settings = {
        "name": "tiny",
        "suite": "classical",
        "functions": ["shifted_sphere", "rosenbrock", "shifted_ackley"],
        "dim": 4,
        "algorithms": ["ECO", "EECO", "DECO", "EDECO"],
        "pop_size": 12,
        "max_fes": 400,
        "runs": 3,
        "seed": 1,
        "workers": 1,
        "history_points": 10,
    }
    settings.update(overrides)
    return ExperimentConfig(**settings)


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_checkpoints_and_history_sampling():
    checkpoints = checkpoint_evaluations(1000, 5)
    assert list(checkpoints) == [1, 251, 500, 750, 1000]
    import numpy as np

    history = np.arange(1000, 0, -1.0)
    assert sample_history(history, checkpoints) == [1000.0, 750.0, 501.0, 251.0, 1.0]


def test_tasks_share_the_seed_across_algorithms_but_not_across_runs():
    tasks = build_tasks(tiny_config())
    assert len(tasks) == 3 * 3 * 4
    by_key = {}
    for task in tasks:
        by_key.setdefault((task["function"], task["run"]), set()).add(task["seed"])
    assert all(len(seeds) == 1 for seeds in by_key.values())
    assert len({next(iter(seeds)) for seeds in by_key.values()}) == 9


def test_runner_writes_raw_results_and_is_deterministic(tmp_path):
    first = run_experiment(tiny_config(), tmp_path / "a")
    second = run_experiment(tiny_config(workers=2), tmp_path / "b")
    rows_a, rows_b = read_csv(first / "runs.csv"), read_csv(second / "runs.csv")
    assert len(rows_a) == 36
    for row_a, row_b in zip(rows_a, rows_b, strict=True):
        assert row_a["n_evals"] == "400"
        for column in ("function", "algorithm", "run", "seed", "best_f", "error"):
            assert row_a[column] == row_b[column]  # identical with 1 or 2 workers
    records = stats.load_records(first)
    assert len(records[0]["history"]) == len(records[0]["history_evals"]) == 10
    assert records[0]["history"][-1] == pytest.approx(records[0]["best_f"], rel=1e-9)
    assert len(records[0]["best_x"]) == 4
    metadata = json.loads((first / "config.json").read_text())
    assert metadata["n_runs_total"] == 36 and metadata["config"]["max_fes"] == 400
    assert set(metadata["algorithms"]) == {"ECO", "EECO", "DECO", "EDECO"}


def test_tables_and_figures_are_generated(tmp_path):
    results_dir = run_experiment(tiny_config(), tmp_path / "results")
    tables = write_all_tables(results_dir)
    assert {path.name for path in tables} == {
        "summary.md",
        "friedman.md",
        "rank_sum.md",
        "rank_sum_detail.md",
        "rank_sum_vs_baseline.md",
    }
    summary = (results_dir / "tables" / "summary.md").read_text()
    assert "| Function | Index | ECO | EECO | DECO | EDECO |" in summary
    assert summary.count("| Mean |") == 3
    rank_sum = read_csv(results_dir / "tables" / "rank_sum.csv")
    assert [row["EDECO vs."] for row in rank_sum] == ["ECO", "EECO", "DECO"]
    assert all(int(r["+"]) + int(r["="]) + int(r["-"]) == 3 for r in rank_sum)
    baseline = read_csv(results_dir / "tables" / "rank_sum_vs_baseline.csv")
    assert [row["vs. ECO"] for row in baseline] == ["EECO", "DECO", "EDECO"]
    # Table 4 and Table 5 layouts are mirror images for the pair (ECO, EDECO).
    assert baseline[2]["+"] == rank_sum[0]["+"] and baseline[2]["-"] == rank_sum[0]["-"]
    figures = write_all_figures(results_dir, dpi=50)
    assert len(figures) == 4
    assert all(path.exists() and path.stat().st_size > 2000 for path in figures)


def test_engineering_experiment_reports_feasibility(tmp_path):
    config = tiny_config(
        suite="engineering",
        functions=["RW03", "RW07"],
        dim=None,
        algorithms={"ECO": {}, "EDECO-small": {"base": "EDECO", "pop_size": 10}},
        max_fes=600,
        runs=2,
    )
    results_dir = run_experiment(config, tmp_path / "eng")
    rows = read_csv(results_dir / "runs.csv")
    assert len(rows) == 8
    assert {row["pop_size"] for row in rows if row["algorithm"] == "EDECO-small"} == {"10"}
    assert all(row["feasible"] in ("True", "False") for row in rows)
    assert all(float(row["max_violation"]) >= 0.0 for row in rows)
    write_all_tables(results_dir)
    table = (results_dir / "tables" / "engineering.md").read_text()
    assert "| RW03 | Feasible |" in table and "Friedman mean rank" in table
    # RW07 has a negative objective: the figures must cope with mixed-sign values.
    assert len(write_all_figures(results_dir, dpi=40)) == 4


def test_yaml_configuration_round_trip(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(
        "suite: classical\nfunctions: [sphere]\ndim: 3\nalgorithms: [ECO]\n"
        "pop_size: 8\nmax_fes: 100\nruns: 2\n"
    )
    config = load_config(path, runs=4, workers=None)
    assert config.name == "experiment" and config.runs == 4 and config.workers == 1
    assert config.budget(3) == 100
    assert ExperimentConfig(name="x", suite="classical", dim=7).budget(7) == 35_000
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="classical")  # dim missing
    with pytest.raises(ValueError):
        ExperimentConfig(name="x", suite="unknown", dim=3)


def test_cli_smoke(tmp_path, capsys):
    assert main(["demo", "--dim", "4", "--runs", "1", "--max-fes", "300", "--pop-size", "12"]) == 0
    assert "shifted_sphere" in capsys.readouterr().out

    assert main([
        "solve", "--suite", "engineering", "--function", "RW03", "--algorithm", "EDECO",
        "--pop-size", "12", "--max-fes", "500", "--seed", "3",
    ]) == 0  # fmt: skip
    output = capsys.readouterr().out
    assert "max violation" in output and "evaluations  : 500" in output

    config = tmp_path / "cli.yaml"
    config.write_text(
        "suite: classical\nfunctions: [sphere, shifted_sphere]\ndim: 3\n"
        "algorithms: [ECO, EDECO]\npop_size: 8\nmax_fes: 200\nruns: 2\n"
    )
    out_dir = tmp_path / "out"
    assert main(["run", "--config", str(config), "--output", str(out_dir), "--runs", "3"]) == 0
    assert len(read_csv(out_dir / "runs.csv")) == 2 * 2 * 3
    assert main(["tables", "--results", str(out_dir)]) == 0
    assert main(["figures", "--results", str(out_dir), "--dpi", "40"]) == 0
    assert (out_dir / "tables" / "friedman.md").exists()
    assert (out_dir / "figures" / "convergence.png").exists()
    with pytest.raises(SystemExit):
        main(["solve", "--algorithm", "PSO"])


def test_figures_with_more_algorithms_than_colour_slots(tmp_path):
    variants = {f"ECO-{k}": {"base": "ECO", "levy_beta": 1.1 + 0.1 * k} for k in range(9)}
    config = tiny_config(
        functions=["sphere", "shifted_sphere"], algorithms=variants, max_fes=120, runs=2
    )
    results_dir = run_experiment(config, tmp_path / "many")
    figures = write_all_figures(results_dir, dpi=40)
    # Convergence curves and box plots are skipped instead of inventing extra colours.
    assert sorted(path.name for path in figures) == ["friedman_ranks.png", "rank_sum.png"]
