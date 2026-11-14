"""Tests of the dynamic fitness-distance balancing strategy (Eqs. 16-18)."""

from __future__ import annotations

import numpy as np
import pytest

from edeco.dfs import dfs_scores, dfs_weight, select_dfs


def test_sawtooth_weight_range_and_period():
    t_max, alpha, beta = 500, 10.0, 0.4
    weights = np.array([dfs_weight(t, t_max, alpha, beta) for t in range(1, t_max + 1)])
    assert weights.min() == pytest.approx(beta)
    assert beta <= weights.min() and weights.max() < 1.0
    period = 50  # t_max / alpha
    # The weight restarts from beta at every multiple of the period ...
    assert all(
        dfs_weight(k * period, t_max, alpha, beta) == pytest.approx(beta) for k in (1, 5, 10)
    )
    # ... and grows linearly towards 1 inside a period.
    assert dfs_weight(25, t_max, alpha, beta) == pytest.approx(0.4 + 0.6 * 0.5)
    inside = weights[: period - 1]
    assert np.all(np.diff(inside) > 0.0)
    np.testing.assert_allclose(weights[:period], weights[period : 2 * period])


def test_literal_weight_matches_eq_18_as_printed_and_is_nearly_constant():
    t_max, alpha, beta = 500, 10.0, 0.4
    # t > t_max / alpha: mod(t_max / alpha, t) = t_max / alpha.
    assert dfs_weight(200, t_max, alpha, beta, "literal") == pytest.approx(0.1 * 0.6 + 0.4)
    # t = 7: mod(50, 7) = 1.
    assert dfs_weight(7, t_max, alpha, beta, "literal") == pytest.approx(1 / 500 * 0.6 + 0.4)
    weights = [dfs_weight(t, t_max, alpha, beta, "literal") for t in range(1, t_max + 1)]
    assert min(weights) >= beta and max(weights) <= beta + (1 - beta) / alpha + 1e-12


def test_unknown_schedule_is_rejected():
    with pytest.raises(ValueError):
        dfs_weight(1, 10, schedule="cosine")


def test_scores_match_a_hand_computed_example():
    population = np.array([[0.0, 0.0], [3.0, 4.0], [6.0, 8.0], [0.0, 1.0]])
    fitness = np.array([1.0, 2.0, 5.0, 3.0])  # best individual: index 0
    # Euclidean distances to the best (Eq. 17): 0, 5, 10, 1 -> normalised 0, .5, 1, .1
    # Fitness normalised with the best at 1 and the worst at 0: 1, .75, 0, .5
    weight = 0.6
    expected = weight * np.array([1.0, 0.75, 0.0, 0.5]) + (1 - weight) * np.array(
        [0.0, 0.5, 1.0, 0.1]
    )
    np.testing.assert_allclose(dfs_scores(population, fitness, weight), expected)
    assert select_dfs(population, fitness, weight) == 1  # fit and reasonably far away


def test_weight_one_selects_the_best_and_weight_zero_the_farthest():
    rng = np.random.default_rng(0)
    population = rng.normal(size=(20, 5))
    fitness = rng.random(20)
    best = int(np.argmin(fitness))
    assert select_dfs(population, fitness, 1.0) == best
    farthest = int(np.argmax(np.linalg.norm(population - population[best], axis=1)))
    assert select_dfs(population, fitness, 0.0) == farthest
    # The best individual always scores exactly the fitness weight.
    assert dfs_scores(population, fitness, 0.7)[best] == pytest.approx(0.7)


def test_degenerate_populations_do_not_produce_nan():
    identical = np.ones((6, 3))
    scores = dfs_scores(identical, np.full(6, 2.0), 0.5)
    assert np.all(np.isfinite(scores))
    assert select_dfs(identical, np.full(6, 2.0), 0.5) == 0  # deterministic tie-break
    # Equal fitness but distinct positions: the farthest individual is selected.
    population = np.array([[0.0], [1.0], [5.0]])
    assert select_dfs(population, np.full(3, 2.0), 0.5) == 2


def test_infinite_fitness_is_treated_as_the_worst_value():
    population = np.array([[0.0], [1.0], [2.0]])
    fitness = np.array([1.0, np.inf, 3.0])
    scores = dfs_scores(population, fitness, 0.9)
    assert np.all(np.isfinite(scores))
    assert int(np.argmax(scores)) == 0
