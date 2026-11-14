"""Result tables in the layout of the paper, written as Markdown and CSV.

* ``summary_table``: best / mean / std per function (Tables A1-A4),
* ``friedman_table``: Friedman mean ranks and p-value (Tables 3 and 6),
* ``rank_sum_baseline_table``: Wilcoxon rank-sum ``+ / = / -`` counts of every variant
  against the baseline ECO (Table 4),
* ``rank_sum_table``: counts of EDECO against every other algorithm (Table 5),
* ``engineering_table``: engineering design problems (Table 8).
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats as scipy_stats

from edeco.analysis import stats

Table = tuple[list[str], list[list[str]]]


def _fmt(value: float) -> str:
    return "n/a" if value is None or math.isnan(value) else f"{value:.4E}"


def to_markdown(table: Table) -> str:
    """Render ``(header, rows)`` as a GitHub-flavoured Markdown table."""
    header, rows = table
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(["---"] * len(header)) + "|"]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def write_table(table: Table, path: Path) -> None:
    """Write a table as ``<path>.md`` and ``<path>.csv``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".md").write_text(to_markdown(table), encoding="utf-8")
    with open(path.with_suffix(".csv"), "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(table[0])
        writer.writerows(table[1])


def summary_table(results: stats.Results) -> Table:
    """Best, mean, standard deviation and rank (by mean) per function and algorithm."""
    summary = stats.summarize(results)
    ranks = stats.rank_rows(results.matrix("mean"))
    header = ["Function", "Index", *results.algorithms]
    rows = []
    for i, function in enumerate(results.functions):
        for label, key in (("Best", "best"), ("Mean", "mean"), ("Std", "std")):
            rows.append(
                [function, label, *[_fmt(summary[function][a][key]) for a in results.algorithms]]
            )
        rows.append([function, "Rank", *[f"{r:g}" for r in ranks[i]]])
    return header, rows


def friedman_table(results: stats.Results) -> Table:
    """Friedman mean ranks of the algorithms and the p-value of the Friedman test."""
    outcome = stats.friedman_test(results.matrix("mean"))
    order = scipy_stats.rankdata(outcome.mean_ranks, method="min")
    header = ["Algorithm", "Friedman mean rank", "Overall rank"]
    rows = [
        [algorithm, f"{outcome.mean_ranks[i]:.4f}", str(int(order[i]))]
        for i, algorithm in enumerate(results.algorithms)
    ]
    rows.append(["p-value", f"{outcome.p_value:.2E}", ""])
    if 2 <= len(results.algorithms) <= 10:
        cd = stats.nemenyi_critical_difference(len(results.algorithms), outcome.n_functions)
        rows.append(["Nemenyi CDV (alpha = 0.05)", f"{cd:.4f}", ""])
    return header, rows


def rank_sum_table(results: stats.Results, reference: str, alpha: float = 0.05) -> Table:
    """Wilcoxon rank-sum ``+ / = / -`` counts of ``reference`` against the others.

    ``+`` means that ``reference`` is significantly better on a function, ``-`` that it is
    significantly worse and ``=`` that the difference is not significant at ``alpha``.
    """
    counts = stats.rank_sum_counts(results, reference, alpha)
    header = [f"{reference} vs.", "+", "=", "-", "+/=/-"]
    rows = [
        [name, str(c["+"]), str(c["="]), str(c["-"]), f"{c['+']}/{c['=']}/{c['-']}"]
        for name, c in counts.items()
    ]
    return header, rows


def rank_sum_baseline_table(results: stats.Results, baseline: str, alpha: float = 0.05) -> Table:
    """Wilcoxon rank-sum ``+ / = / -`` counts of every algorithm against ``baseline``.

    This is the layout of Table 4 of the paper ("vs. ECO"): ``+`` means that the
    algorithm in the row is significantly better than the baseline.
    """
    counts = stats.rank_sum_counts_against(results, baseline, alpha)
    header = [f"vs. {baseline}", "+", "=", "-", "+/=/-"]
    rows = [
        [name, str(c["+"]), str(c["="]), str(c["-"]), f"{c['+']}/{c['=']}/{c['-']}"]
        for name, c in counts.items()
    ]
    return header, rows


def rank_sum_detail_table(results: stats.Results, reference: str, alpha: float = 0.05) -> Table:
    """Per-function rank-sum outcome and p-value of ``reference`` against the others."""
    others = [a for a in results.algorithms if a != reference]
    header = ["Function", *[f"vs. {a}" for a in others]]
    rows = []
    for function in results.functions:
        row = [function]
        for algorithm in others:
            outcome, p_value = stats.rank_sum_outcome(
                results.values[function][reference], results.values[function][algorithm], alpha
            )
            row.append(f"{outcome} (p = {p_value:.2E})")
        rows.append(row)
    return header, rows


def engineering_table(results_dir: str | Path) -> Table:
    """Engineering problems: objective of the feasible runs, feasible count and rank.

    ``Best``, ``Mean`` and ``Std`` are computed over the design objective of the runs
    whose returned solution is feasible (``n/a`` when no run is feasible); ``Feasible``
    gives the number of such runs.  ``Rank`` orders the algorithms by the mean of the
    penalised objective over all runs, so an algorithm that returns infeasible designs is
    ranked behind those that do not.
    """
    records = stats.load_records(results_dir)
    penalised = stats.load_results(results_dir, metric="best_f")
    ranks = stats.rank_rows(penalised.matrix("mean"))
    objective: dict[tuple[str, str], list[float]] = {}
    n_runs: dict[tuple[str, str], int] = {}
    for record in records:
        key = (record["function"], record["algorithm"])
        n_runs[key] = n_runs.get(key, 0) + 1
        objective.setdefault(key, [])
        if record["feasible"]:
            objective[key].append(float(record["objective"]))
    reducers = (("Best", np.min), ("Mean", np.mean), ("Std", stats.sample_std))
    header = ["Problem", "Index", *penalised.algorithms]
    rows = []
    for i, function in enumerate(penalised.functions):
        for label, reducer in reducers:
            row = [function, label]
            for algorithm in penalised.algorithms:
                values = np.array(objective[(function, algorithm)])
                row.append(_fmt(float(reducer(values))) if values.size else "n/a")
            rows.append(row)
        rows.append(
            [
                function,
                "Feasible",
                *[
                    f"{len(objective[(function, a)])}/{n_runs[(function, a)]}"
                    for a in penalised.algorithms
                ],
            ]
        )
        rows.append([function, "Rank", *[f"{r:g}" for r in ranks[i]]])
    rows.append(["All", "Friedman mean rank", *[f"{r:.2f}" for r in ranks.mean(axis=0)]])
    return header, rows


def write_all_tables(results_dir: str | Path, reference: str | None = None) -> list[Path]:
    """Compute every table that applies to the results in ``results_dir``.

    Tables are written to ``<results_dir>/tables``.  ``reference`` is the algorithm the
    rank-sum test compares against the others (default: ``EDECO`` when present, else the
    last algorithm).  The first algorithm of the experiment is taken as the baseline of
    the Table 4 layout.
    """
    results_dir = Path(results_dir)
    out_dir = results_dir / "tables"
    results = stats.load_results(results_dir)
    if reference is None:
        reference = "EDECO" if "EDECO" in results.algorithms else results.algorithms[-1]
    written = []
    tables = {
        "summary": summary_table(results),
        "friedman": friedman_table(results),
        "rank_sum": rank_sum_table(results, reference),
        "rank_sum_detail": rank_sum_detail_table(results, reference),
    }
    if results.algorithms[0] != reference:
        tables["rank_sum_vs_baseline"] = rank_sum_baseline_table(results, results.algorithms[0])
    records = stats.load_records(results_dir)
    if records and records[0]["suite"] == "engineering":
        tables["engineering"] = engineering_table(results_dir)
    for name, table in tables.items():
        write_table(table, out_dir / name)
        written.append(out_dir / f"{name}.md")
    return written
