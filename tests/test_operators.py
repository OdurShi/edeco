"""Equation-level tests of the ECO operators (Eqs. 1-11 and 19)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edeco import operators as ops


def test_clamp_sets_out_of_range_coordinates_to_the_bound():
    lb, ub = np.array([-1.0, 0.0, 2.0]), np.array([1.0, 5.0, 3.0])
    assert np.array_equal(ops.clamp(np.array([-4.0, 2.5, 9.0]), lb, ub), [-1.0, 2.5, 3.0])


def test_logistic_map_init_shape_bounds_and_formula():
    lb, ub = np.array([-5.0, 0.0, 10.0]), np.array([5.0, 1.0, 20.0])
    population = ops.logistic_map_init(np.random.default_rng(3), 50, lb, ub)
    assert population.shape == (50, 3)
    assert np.all(population >= lb) and np.all(population <= ub)
    # Eq. (1)-(2) with the reference semantics: one logistic step from a uniform seed.
    seed_values = np.random.default_rng(3).random((50, 3))
    expected = lb + (ub - lb) * 4.0 * seed_values * (1.0 - seed_values)
    np.testing.assert_allclose(population, expected)


def test_logistic_map_init_is_deterministic_for_a_seed():
    lb, ub = np.full(4, -100.0), np.full(4, 100.0)
    first = ops.logistic_map_init(np.random.default_rng(11), 8, lb, ub)
    second = ops.logistic_map_init(np.random.default_rng(11), 8, lb, ub)
    other = ops.logistic_map_init(np.random.default_rng(12), 8, lb, ub)
    assert np.array_equal(first, second)
    assert not np.array_equal(first, other)


def test_levy_step_shape_determinism_and_heavy_tail():
    step = ops.levy_step(np.random.default_rng(0), 7)
    assert step.shape == (7,) and np.all(np.isfinite(step))
    assert np.array_equal(step, ops.levy_step(np.random.default_rng(0), 7))
    sample = ops.levy_step(np.random.default_rng(1), 200_000)
    assert abs(np.median(sample)) < 0.02  # symmetric around zero
    # Heavy tails: far more mass beyond 10 than a unit normal could ever have.
    assert np.mean(np.abs(sample) > 10.0) > 1e-3


def test_step_weight_matches_eq_3():
    assert ops.step_weight(0.0) == pytest.approx(0.1 * math.log(2.0))
    assert ops.step_weight(0.5) == pytest.approx(0.1 * math.log(1.5))
    assert ops.step_weight(1.0) == 0.0


def test_amplitude_matches_eq_8():
    assert ops.amplitude(0.25, 0.5) == pytest.approx(4.0 * 0.5 * 0.75)
    assert ops.amplitude(1.0, 3.0) == 0.0


def test_talent_product_matches_eq_9():
    # R > H: E = pi / P * t / t_max, hence P * E = pi * t / t_max.
    assert ops.talent_product(0.3, p=1.7, r=0.9, h=0.5) == pytest.approx(math.pi * 0.3)
    # R <= H: E = 1, hence P * E = P.
    assert ops.talent_product(0.3, p=1.7, r=0.5, h=0.5) == 1.7
    assert ops.talent_product(0.3, p=-2.0, r=0.1, h=0.5) == -2.0


def test_nearest_school_uses_manhattan_distance():
    schools = np.array([[3.0, 0.0], [2.0, 2.0]])
    students = np.array([[0.0, 0.0]])
    # L1 distances are 3 and 4 (school 0 wins) although the Euclidean distances are
    # 3 and 2.83 (school 1 would win).
    assert np.array_equal(ops.nearest_school(students, schools), [[3.0, 0.0]])


def test_nearest_school_shape_and_tie_break():
    schools = np.array([[1.0, 0.0], [-1.0, 0.0], [5.0, 5.0]])
    students = np.array([[0.0, 0.0], [4.0, 4.0], [-3.0, 0.0], [0.0, 7.0]])
    close = ops.nearest_school(students, schools)
    assert close.shape == students.shape
    assert np.array_equal(close[0], schools[0])  # equidistant: lowest index is kept
    assert np.array_equal(close[1], schools[2])
    assert np.array_equal(close[2], schools[1])


def test_primary_school_update_uses_the_agents_own_mean():
    x = np.array([1.0, 2.0, 6.0])  # mean of its coordinates = 3
    levy = np.array([1.0, -2.0, 0.5])
    expected = x + 0.05 * (3.0 - x) * levy
    np.testing.assert_allclose(ops.primary_school_update(x, 0.05, levy), expected)


def test_primary_student_update_moves_along_the_line_to_the_school():
    x, close = np.array([0.0, 0.0]), np.array([10.0, -10.0])
    np.testing.assert_allclose(ops.primary_student_update(x, 0.1, close, 2.0), [2.0, -2.0])
    np.testing.assert_allclose(ops.primary_student_update(x, 0.1, close, 0.0), x)


def test_middle_school_update_matches_eq_6():
    x = np.array([1.0, 1.0])
    best, mean = np.array([3.0, 5.0]), np.array([1.0, 2.0])
    levy = np.array([0.5, -1.0])
    expected = x + (best - mean) * math.exp(0.4 - 1.0) * levy
    np.testing.assert_allclose(ops.middle_school_update(x, best, mean, 0.4, levy), expected)


@pytest.mark.parametrize("use_talent", [True, False])
def test_middle_student_update_matches_eq_7(use_talent):
    x, close = np.array([2.0, -1.0, 4.0]), np.array([1.0, 1.0, 1.0])
    w, p, tau = 0.06, 1.3, 0.35
    e = math.pi / p * tau if use_talent else 1.0
    c = ops.talent_product(tau, p, 0.9 if use_talent else 0.1, 0.5)
    expected = x - w * close - p * (e * w * close - x)
    np.testing.assert_allclose(ops.middle_student_update(x, w, close, p, c), expected)


def test_high_school_update_reference_and_paper_anchor():
    x, best, mean = np.array([1.0, 2.0]), np.array([4.0, 6.0]), np.array([0.0, 1.0])
    reference = ops.high_school_update(x, best, x, 0.5, -0.25)
    np.testing.assert_allclose(reference, x + (best - x) * 0.5 - (best - x) * (-0.25))
    paper = ops.high_school_update(x, best, mean, 0.5, -0.25)
    np.testing.assert_allclose(paper, x + (best - mean) * 0.5 - (best - mean) * (-0.25))
    # Equal random draws cancel: the school does not move.
    np.testing.assert_allclose(ops.high_school_update(x, best, x, 0.7, 0.7), x)


def test_high_student_update_matches_eq_11_in_both_forms():
    x, best = np.array([1.0, -2.0]), np.array([3.0, 3.0])
    p, tau = -0.8, 0.6
    e = math.pi / p * tau
    c = ops.talent_product(tau, p, 0.9, 0.5)
    reference = ops.high_student_update(x, best, best, p, c)  # reference ECO code
    np.testing.assert_allclose(reference, best - p * (e * best - x))
    printed = ops.high_student_update(x, best, x, p, c)  # Eq. (11) as printed
    np.testing.assert_allclose(printed, x - p * (e * best - x))


def test_dfs_student_update_is_eq_7_with_the_leading_term_replaced():
    x, close, x_dfs = np.array([2.0, -1.0]), np.array([1.0, 1.0]), np.array([7.0, 7.0])
    w, p, c = 0.05, 0.9, 1.1
    expected = ops.middle_student_update(x, w, close, p, c) - x + x_dfs
    np.testing.assert_allclose(ops.dfs_student_update(x, x_dfs, w, close, p, c), expected)


def test_updates_reduce_to_their_anchor_at_the_end_of_the_run():
    """At t = t_max, w = P = 0: only the talent term (pi * w * close = 0) remains."""
    x, close, x_dfs = np.array([2.0, -1.0]), np.array([1.0, 1.0]), np.array([7.0, 7.0])
    w, p = ops.step_weight(1.0), ops.amplitude(1.0, 1.234)
    np.testing.assert_allclose(ops.dfs_student_update(x, x_dfs, w, close, p, p), x_dfs)
    np.testing.assert_allclose(ops.middle_student_update(x, w, close, p, p), x)
