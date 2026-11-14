"""Tests of the estimation-of-distribution step (Eqs. 12-15) and its selection."""

from __future__ import annotations

import numpy as np
import pytest

from edeco.eda import (
    fit_gaussian,
    pairwise_replace,
    select_dominant,
    truncate_union,
)


def test_fit_gaussian_matches_eq_13_and_14():
    rng = np.random.default_rng(0)
    dominant = rng.normal(size=(12, 4)) * np.array([1.0, 5.0, 0.2, 3.0]) + 7.0
    model = fit_gaussian(dominant)
    np.testing.assert_allclose(model.mean, dominant.mean(axis=0))
    # Eq. (14) normalises by N_d (not N_d - 1).
    np.testing.assert_allclose(model.covariance, np.cov(dominant, rowvar=False, bias=True))
    assert model.n_dominant == 12


def test_fit_gaussian_recovers_a_known_distribution():
    """Statistical sanity: a large sample of a known Gaussian gives back its parameters."""
    rng = np.random.default_rng(1)
    mean = np.array([1.0, -2.0, 0.5])
    factor = np.array([[2.0, 0.0, 0.0], [0.6, 1.0, 0.0], [-0.4, 0.3, 0.5]])
    covariance = factor @ factor.T
    population = mean + rng.standard_normal((20_000, 3)) @ factor.T
    model = fit_gaussian(population)
    np.testing.assert_allclose(model.mean, mean, atol=0.05)
    np.testing.assert_allclose(model.covariance, covariance, atol=0.12)


def test_samples_follow_the_fitted_model():
    rng = np.random.default_rng(2)
    dominant = rng.normal(size=(15, 3)) @ np.array([[3.0, 0, 0], [1.0, 1.0, 0], [0, 0.5, 0.2]])
    model = fit_gaussian(dominant + np.array([10.0, -5.0, 0.0]))
    samples = model.sample(100_000, np.random.default_rng(3))
    assert samples.shape == (100_000, 3)
    np.testing.assert_allclose(samples.mean(axis=0), model.mean, atol=0.05)
    np.testing.assert_allclose(np.cov(samples, rowvar=False), model.covariance, atol=0.15)


def test_sampling_is_deterministic_for_a_seed():
    model = fit_gaussian(np.random.default_rng(4).normal(size=(6, 5)))
    first = model.sample(9, np.random.default_rng(8))
    second = model.sample(9, np.random.default_rng(8))
    assert np.array_equal(first, second)


def test_rank_deficient_model_samples_stay_in_the_span_of_the_dominant_group():
    """With N_d <= D the covariance is singular; sampling must still work (Eq. 15)."""
    rng = np.random.default_rng(5)
    dominant = rng.normal(size=(4, 10))  # rank of the covariance is at most 3
    model = fit_gaussian(dominant)
    assert np.linalg.matrix_rank(model.covariance) == 3
    samples = model.sample(200, np.random.default_rng(6))
    assert np.all(np.isfinite(samples))
    # Components orthogonal to the span of the centred dominant group vanish.
    basis = np.linalg.svd(model.deviations, full_matrices=False)[2][:3]
    centred = samples - model.mean
    residual = centred - centred @ basis.T @ basis
    assert np.max(np.abs(residual)) < 1e-10


def test_fit_gaussian_rejects_an_empty_group():
    with pytest.raises(ValueError):
        fit_gaussian(np.empty((0, 3)))


def test_select_dominant_returns_the_fittest_with_stable_ties():
    fitness = np.array([5.0, 1.0, 3.0, 1.0, 0.5])
    assert np.array_equal(select_dominant(fitness, 3), [4, 1, 3])


def test_truncate_union_keeps_the_best_n_sorted():
    population = np.array([[0.0], [1.0], [2.0]])
    fitness = np.array([1.0, 4.0, 9.0])
    samples = np.array([[10.0], [11.0]])
    sample_fitness = np.array([0.5, 20.0])
    new_population, new_fitness, survivors = truncate_union(
        population, fitness, samples, sample_fitness
    )
    assert np.array_equal(new_fitness, [0.5, 1.0, 4.0])
    assert np.array_equal(new_population, [[10.0], [0.0], [1.0]])
    assert survivors == 1


def test_truncate_union_prefers_incumbents_on_ties():
    population = np.array([[0.0], [1.0]])
    fitness = np.array([2.0, 2.0])
    samples = np.array([[5.0]])
    new_population, _, survivors = truncate_union(population, fitness, samples, np.array([2.0]))
    assert np.array_equal(new_population, population)
    assert survivors == 0


def test_pairwise_replace_only_replaces_when_not_worse():
    population = np.array([[0.0], [1.0], [2.0]])
    fitness = np.array([1.0, 4.0, 9.0])
    samples = np.array([[10.0], [11.0]])
    new_population, new_fitness, survivors = pairwise_replace(
        population, fitness, samples, np.array([3.0, 5.0]), targets=np.array([0, 2])
    )
    # Sample 0 (3.0) loses against individual 0 (1.0); sample 1 (5.0) beats individual 2.
    assert np.array_equal(new_fitness, [1.0, 4.0, 5.0])
    assert np.array_equal(new_population, [[0.0], [1.0], [11.0]])
    assert survivors == 1
    assert np.array_equal(fitness, [1.0, 4.0, 9.0])  # inputs are not modified
