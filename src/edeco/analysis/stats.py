"""Statistics used in the paper's evaluation (Sections 4.2-4.4).

* per-function summary (best / mean / standard deviation, Tables A1-A4 and 8),
* Friedman ranking and test (Tables 3 and 6, Fig. 6 and 8),
* Wilcoxon rank-sum test at the 0.05 level with ``+ / = / -`` counts (Tables 4 and 5),
* critical difference of the Nemenyi post-hoc test (Eq. 20, Figs. 3 and 7).
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

# Critical values q_alpha of the Nemenyi test (studentised range statistic divided by
# sqrt(2)) for k = 2 ... 10 compared algorithms; Demsar (2006), JMLR 7, Table 5.
_NEMENYI_Q = {
    0.05: (1.960, 2.343, 2.569, 2.728, 2.850, 2.949, 3.031, 3.102, 3.164),
    0.10: (1.645, 2.052, 2.291, 2.459, 2.589, 2.693, 2.780, 2.855, 2.920),
}


@dataclass(frozen=True)
class Results:
    """Final values of an experiment arranged as ``values[function][algorithm]``.

    ``values`` holds, for each function and algorithm, the array of final results of the
    independent runs (the error ``f - f*`` when the optimum is known, else ``f``).
    """

    functions: list[str]
    algorithms: list[str]
    values: dict[str, dict[str, np.ndarray]]
    metric: str

    def matrix(self, statistic: str = "mean") -> np.ndarray:
        """``(n_functions, n_algorithms)`` matrix of a per-function statistic."""
        reducers = {"mean": np.mean, "median": np.median, "best": np.min, "std": sample_std}
        reducer = reducers[statistic]
        return np.array(
            [[reducer(self.values[f][a]) for a in self.algorithms] for f in self.functions]
        )


def sample_std(values: np.ndarray) -> float:
    """Sample standard deviation (``ddof=1``, as MATLAB's ``std``); 0 for a single run."""
    return float(np.std(values, ddof=1)) if values.size > 1 else 0.0


def load_results(results_dir: str | Path, metric: str | None = None) -> Results:
    """Read ``runs.csv`` written by :func:`edeco.experiments.runner.run_experiment`.

    ``metric`` selects the column: ``"error"`` (default when available), ``"best_f"`` or
    ``"objective"`` (engineering problems: design objective of the returned solution).
    """
    with open(Path(results_dir) / "runs.csv", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no runs found in {results_dir}")
    if metric is None:
        metric = "error" if rows[0]["error"] != "" else "best_f"
    functions: list[str] = []
    algorithms: list[str] = []
    collected: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        function, algorithm = row["function"], row["algorithm"]
        if function not in collected:
            functions.append(function)
            collected[function] = {}
        if algorithm not in algorithms:
            algorithms.append(algorithm)
        collected[function].setdefault(algorithm, []).append(float(row[metric]))
    values = {f: {a: np.array(v) for a, v in by_alg.items()} for f, by_alg in collected.items()}
    return Results(functions=functions, algorithms=algorithms, values=values, metric=metric)


def load_records(results_dir: str | Path) -> list[dict]:
    """Read the full per-run records (``runs.jsonl``)."""
    with open(Path(results_dir) / "runs.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(results: Results) -> dict[str, dict[str, dict[str, float]]]:
    """Best, mean, standard deviation, median and worst per function and algorithm."""
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for function in results.functions:
        summary[function] = {}
        for algorithm in results.algorithms:
            runs = results.values[function][algorithm]
            summary[function][algorithm] = {
                "best": float(np.min(runs)),
                "mean": float(np.mean(runs)),
                "std": sample_std(runs),
                "median": float(np.median(runs)),
                "worst": float(np.max(runs)),
                "runs": int(runs.size),
            }
    return summary


def rank_rows(matrix: np.ndarray) -> np.ndarray:
    """Rank the algorithms on every function (1 = best; ties receive average ranks)."""
    return np.vstack([stats.rankdata(row, method="average") for row in matrix])


@dataclass(frozen=True)
class FriedmanResult:
    """Outcome of the Friedman test over functions (blocks) and algorithms."""

    mean_ranks: np.ndarray
    statistic: float
    p_value: float
    n_functions: int


def friedman_test(matrix: np.ndarray) -> FriedmanResult:
    """Friedman ranking and test on a ``(n_functions, n_algorithms)`` matrix.

    The paper applies the test to the mean results of each algorithm on each function.
    With fewer than three algorithms the chi-square test is undefined and ``nan`` is
    returned for the statistic and the p-value.
    """
    ranks = rank_rows(matrix)
    n_functions, n_algorithms = matrix.shape
    if n_algorithms >= 3 and n_functions >= 2:
        statistic, p_value = stats.friedmanchisquare(*matrix.T)
    else:
        statistic, p_value = math.nan, math.nan
    return FriedmanResult(
        mean_ranks=ranks.mean(axis=0),
        statistic=float(statistic),
        p_value=float(p_value),
        n_functions=n_functions,
    )


def nemenyi_critical_difference(n_algorithms: int, n_functions: int, alpha: float = 0.05) -> float:
    """Critical difference value ``CDV = q_alpha * sqrt(K (K + 1) / (6 N_f))``, Eq. (20)."""
    if alpha not in _NEMENYI_Q:
        raise ValueError(f"alpha must be one of {sorted(_NEMENYI_Q)}")
    if not 2 <= n_algorithms <= 10:
        raise ValueError("critical values are tabulated for 2 to 10 algorithms")
    q_alpha = _NEMENYI_Q[alpha][n_algorithms - 2]
    return q_alpha * math.sqrt(n_algorithms * (n_algorithms + 1) / (6.0 * n_functions))


def rank_sum_outcome(
    reference: Sequence[float], other: Sequence[float], alpha: float = 0.05
) -> tuple[str, float]:
    """Wilcoxon rank-sum comparison of two samples of final results (minimisation).

    Returns ``("+", p)`` when ``reference`` is significantly better (smaller) than
    ``other``, ``("-", p)`` when it is significantly worse and ``("=", p)`` otherwise.
    The two-sided test is ``scipy.stats.mannwhitneyu`` (the Mann-Whitney U test is the
    Wilcoxon rank-sum test); the direction is given by the mean ranks of the two samples.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(other, dtype=float)
    if np.all(a == a[0]) and np.all(b == a[0]):
        return "=", 1.0
    statistic, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    if not p_value < alpha:
        return "=", float(p_value)
    # U counts pairs with a > b (plus half the ties): a small U means `reference` is smaller.
    return ("+" if statistic < a.size * b.size / 2.0 else "-"), float(p_value)


def rank_sum_counts(
    results: Results, reference: str, alpha: float = 0.05
) -> dict[str, dict[str, int]]:
    """``+ / = / -`` counts of ``reference`` against every other algorithm."""
    counts: dict[str, dict[str, int]] = {}
    for algorithm in results.algorithms:
        if algorithm == reference:
            continue
        tally = {"+": 0, "=": 0, "-": 0}
        for function in results.functions:
            outcome, _ = rank_sum_outcome(
                results.values[function][reference], results.values[function][algorithm], alpha
            )
            tally[outcome] += 1
        counts[algorithm] = tally
    return counts


def rank_sum_counts_against(
    results: Results, baseline: str, alpha: float = 0.05
) -> dict[str, dict[str, int]]:
    """``+ / = / -`` counts of every other algorithm against ``baseline`` (Table 4 layout).

    ``+`` counts the functions on which the algorithm is significantly better than the
    baseline, ``-`` those on which it is significantly worse.
    """
    counts: dict[str, dict[str, int]] = {}
    for algorithm in results.algorithms:
        if algorithm == baseline:
            continue
        tally = {"+": 0, "=": 0, "-": 0}
        for function in results.functions:
            outcome, _ = rank_sum_outcome(
                results.values[function][algorithm], results.values[function][baseline], alpha
            )
            tally[outcome] += 1
        counts[algorithm] = tally
    return counts
