"""Tests of the statistics behind the paper's tables (Friedman, rank-sum, Nemenyi)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edeco.analysis import stats


def make_results() -> stats.Results:
    rng = np.random.default_rng(0)
    functions = [f"F{i}" for i in range(1, 7)]
    algorithms = ["ECO", "EECO", "EDECO"]
    values = {}
    for function in functions:
        values[function] = {
            "ECO": 10.0 + rng.random(12),
            "EECO": 5.0 + rng.random(12),
            "EDECO": 1.0 + rng.random(12),
        }
    values["F6"]["EECO"] = values["F6"]["EDECO"] + 1e-3 * rng.standard_normal(12)
    return stats.Results(functions=functions, algorithms=algorithms, values=values, metric="error")


def test_rank_rows_assigns_average_ranks_to_ties():
    matrix = np.array([[3.0, 1.0, 2.0], [5.0, 5.0, 1.0]])
    np.testing.assert_allclose(stats.rank_rows(matrix), [[3.0, 1.0, 2.0], [2.5, 2.5, 1.0]])


def test_friedman_mean_ranks_and_p_value():
    results = make_results()
    outcome = stats.friedman_test(results.matrix("mean"))
    assert outcome.n_functions == 6
    np.testing.assert_allclose(outcome.mean_ranks[0], 3.0)  # ECO is always the worst
    assert outcome.mean_ranks[2] < outcome.mean_ranks[1] < outcome.mean_ranks[0]
    assert outcome.mean_ranks.sum() == pytest.approx(6.0)  # 1 + 2 + 3
    assert 0.0 < outcome.p_value < 0.05


def test_friedman_with_two_algorithms_returns_nan_p_value():
    outcome = stats.friedman_test(np.array([[1.0, 2.0], [1.0, 3.0], [2.0, 1.0]]))
    np.testing.assert_allclose(outcome.mean_ranks, [4 / 3, 5 / 3])
    assert math.isnan(outcome.p_value)


def test_nemenyi_critical_difference_is_eq_20():
    # K = 10 algorithms and N_f = 29 functions with q_0.05 = 3.164 (Section 4.3).
    expected = 3.164 * math.sqrt(10 * 11 / (6 * 29))
    assert stats.nemenyi_critical_difference(10, 29) == pytest.approx(expected)
    assert expected == pytest.approx(2.5157, abs=1e-4)
    # K = 4 (ablation study, Fig. 3).
    assert stats.nemenyi_critical_difference(4, 29) == pytest.approx(2.569 * math.sqrt(20 / 174))
    with pytest.raises(ValueError):
        stats.nemenyi_critical_difference(11, 29)
    with pytest.raises(ValueError):
        stats.nemenyi_critical_difference(4, 29, alpha=0.01)


def test_rank_sum_outcomes():
    rng = np.random.default_rng(1)
    low, high = rng.random(15), 5.0 + rng.random(15)
    assert stats.rank_sum_outcome(low, high)[0] == "+"
    assert stats.rank_sum_outcome(high, low)[0] == "-"
    same_a, same_b = rng.random(15), rng.random(15)
    outcome, p_value = stats.rank_sum_outcome(same_a, same_b)
    assert outcome == "=" and p_value >= 0.05
    assert stats.rank_sum_outcome(np.zeros(8), np.zeros(8)) == ("=", 1.0)


def test_rank_sum_counts_against_the_reference():
    counts = stats.rank_sum_counts(make_results(), reference="EDECO")
    assert counts["ECO"] == {"+": 6, "=": 0, "-": 0}
    assert counts["EECO"] == {"+": 5, "=": 1, "-": 0}
    assert "EDECO" not in counts


def test_summarize_reports_sample_standard_deviation():
    results = stats.Results(
        functions=["F1"],
        algorithms=["A"],
        values={"F1": {"A": np.array([1.0, 2.0, 3.0, 6.0])}},
        metric="best_f",
    )
    summary = stats.summarize(results)["F1"]["A"]
    assert summary["best"] == 1.0 and summary["worst"] == 6.0 and summary["mean"] == 3.0
    assert summary["median"] == 2.5 and summary["runs"] == 4
    assert summary["std"] == pytest.approx(np.std([1.0, 2.0, 3.0, 6.0], ddof=1))


def test_rank_sum_counts_against_the_baseline_mirror_the_reference_counts():
    results = make_results()
    against = stats.rank_sum_counts_against(results, baseline="ECO")
    assert against["EDECO"] == {"+": 6, "=": 0, "-": 0}
    assert against["EECO"] == {"+": 6, "=": 0, "-": 0}
    worse = stats.rank_sum_counts_against(results, baseline="EDECO")
    assert worse["ECO"] == {"+": 0, "=": 0, "-": 6}
    assert worse["EECO"] == {"+": 0, "=": 1, "-": 5}
