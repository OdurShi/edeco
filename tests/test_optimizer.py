"""Tests of the evaluation accounting and of the ECO / EECO / DECO / EDECO optimisers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from edeco import (
    ALGORITHMS,
    BudgetExhaustedError,
    ECOConfig,
    Evaluator,
    get_config,
    minimize,
)
from edeco.benchmarks import get_problem


class Recorder:
    """Objective wrapper that counts calls and remembers every evaluated point."""

    def __init__(self, func, lower, upper):
        self.func, self.lower, self.upper = func, lower, upper
        self.calls = 0
        self.values: list[float] = []

    def __call__(self, x):
        assert x.shape == self.lower.shape
        assert np.all(x >= self.lower) and np.all(x <= self.upper), "candidate out of bounds"
        self.calls += 1
        value = self.func(x)
        self.values.append(value)
        return value


def shifted_sphere(x):
    return float(np.sum((x - 30.0) ** 2))


# --------------------------------------------------------------------------- Evaluator


def test_evaluator_counts_tracks_best_and_enforces_the_budget():
    evaluator = Evaluator(lambda x: float(x[0]), max_fes=3)
    assert evaluator(np.array([5.0])) == 5.0
    assert evaluator(np.array([2.0])) == 2.0
    assert evaluator(np.array([4.0])) == 4.0
    assert evaluator.n_evals == 3 and evaluator.exhausted and evaluator.remaining == 0
    assert evaluator.best_f == 2.0 and np.array_equal(evaluator.best_x, [2.0])
    assert np.array_equal(evaluator.history(), [5.0, 2.0, 2.0])
    with pytest.raises(BudgetExhaustedError):
        evaluator(np.array([0.0]))
    assert evaluator.n_evals == 3


def test_evaluator_maps_nan_to_infinity_and_copies_the_best_point():
    evaluator = Evaluator(lambda x: math.nan if x[0] < 0 else float(x[0]), max_fes=5)
    assert evaluator(np.array([-1.0])) == math.inf
    point = np.array([3.0])
    evaluator(point)
    point[0] = 99.0  # must not alter the stored best solution
    assert evaluator.best_f == 3.0 and evaluator.best_x[0] == 3.0


def test_evaluator_rejects_a_non_positive_budget():
    with pytest.raises(ValueError):
        Evaluator(lambda x: 0.0, max_fes=0)


# --------------------------------------------------------------------- budget accounting


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
@pytest.mark.parametrize("max_fes", [7, 20, 333, 1000])
def test_exact_evaluation_accounting(algorithm, max_fes):
    """Every run spends exactly the budget, also when it is not a multiple of N."""
    dim, lower, upper = 5, np.full(5, -100.0), np.full(5, 100.0)
    objective = Recorder(shifted_sphere, lower, upper)
    result = minimize(
        objective,
        (lower, upper),
        dim,
        pop_size=20,
        max_fes=max_fes,
        seed=4,
        config=get_config(algorithm),
    )
    assert objective.calls == max_fes == result.n_evals
    assert result.history.shape == (max_fes,)
    assert np.all(np.diff(result.history) <= 0.0)  # best-so-far never increases
    assert result.best_f == min(objective.values) == result.history[-1]
    assert objective.func(result.best_x) == result.best_f
    assert np.all(result.best_x >= lower) and np.all(result.best_x <= upper)


def test_iteration_horizon_and_eda_bookkeeping():
    kwargs = {"pop_size": 30, "max_fes": 3030, "seed": 0}
    eco = minimize(shifted_sphere, (-100.0, 100.0), 4, config=get_config("ECO"), **kwargs)
    # (3030 - 30) / 30 = 100 iterations of N evaluations each.
    assert eco.n_iterations == eco.info["t_max"] == 100
    assert eco.info["eda_samples"] == 0
    eeco = minimize(shifted_sphere, (-100.0, 100.0), 4, config=get_config("EECO"), **kwargs)
    # N_E = N / 3 = 10 samples per iteration: (3030 - 30) / 40 = 75 iterations.
    assert eeco.n_iterations == eeco.info["t_max"] == 75
    assert eeco.info["eda_samples"] == 750
    assert 0 < eeco.info["eda_survivors"] <= 750


def test_per_dimension_bounds_are_respected():
    lower, upper = np.array([0.0, -5.0, 10.0]), np.array([1.0, 5.0, 20.0])
    objective = Recorder(lambda x: float(np.sum(x**2)), lower, upper)
    result = minimize(
        objective, (lower, upper), 3, pop_size=12, max_fes=600, seed=1, config=get_config("EDECO")
    )
    assert objective.calls == 600
    assert result.best_x[2] >= 10.0


# ------------------------------------------------------------------------- determinism


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_runs_are_reproducible_for_a_seed(algorithm):
    def run(seed):
        return minimize(
            shifted_sphere,
            (-100.0, 100.0),
            6,
            pop_size=20,
            max_fes=1500,
            seed=seed,
            config=get_config(algorithm),
        )

    first, second, other = run(7), run(7), run(8)
    assert first.best_f == second.best_f
    assert np.array_equal(first.best_x, second.best_x)
    assert np.array_equal(first.history, second.history)
    assert not np.array_equal(first.history, other.history)


def test_all_algorithms_start_from_the_same_initial_population():
    """With equal seeds the first N evaluations (the initial population) coincide."""
    histories = [
        minimize(
            shifted_sphere,
            (-100.0, 100.0),
            6,
            pop_size=25,
            max_fes=400,
            seed=3,
            config=get_config(name),
        ).history[:25]
        for name in ALGORITHMS
    ]
    for history in histories[1:]:
        assert np.array_equal(history, histories[0])


def test_the_switches_change_the_search():
    results = {
        name: minimize(
            shifted_sphere,
            (-100.0, 100.0),
            6,
            pop_size=25,
            max_fes=2000,
            seed=3,
            config=get_config(name),
        ).best_f
        for name in ALGORITHMS
    }
    assert len(set(results.values())) == 4


@pytest.mark.parametrize(
    "overrides",
    [
        {"dfs_student_rule": "eq19"},
        {"dfs_weight_schedule": "literal"},
        {"eda_selection": "pairwise"},
        {"base_equations": "paper"},
        {"eda_sample_ratio": 1.0, "dominant_ratio": 0.3},
    ],
)
def test_documented_alternative_readings_run_within_budget(overrides):
    objective = Recorder(shifted_sphere, np.full(5, -100.0), np.full(5, 100.0))
    result = minimize(
        objective,
        (-100.0, 100.0),
        5,
        pop_size=20,
        max_fes=900,
        seed=2,
        config=get_config("EDECO", **overrides),
    )
    assert objective.calls == result.n_evals == 900
    assert math.isfinite(result.best_f)


# --------------------------------------------------------------------------- validation


def test_configuration_and_argument_validation():
    with pytest.raises(ValueError):
        ECOConfig(dfs_student_rule="eq7")
    with pytest.raises(ValueError):
        ECOConfig(dfs_weight_schedule="linear")
    with pytest.raises(ValueError):
        ECOConfig(eda_selection="tournament")
    with pytest.raises(ValueError):
        ECOConfig(base_equations="mine")
    with pytest.raises(ValueError):
        ECOConfig(eda_selection="pairwise", eda_sample_ratio=1.5)
    with pytest.raises(ValueError):
        ECOConfig(dominant_ratio=0.0)
    with pytest.raises(KeyError):
        get_config("PSO")
    with pytest.raises(ValueError):
        minimize(shifted_sphere, (-1.0, 1.0), 3, pop_size=3, max_fes=100)
    with pytest.raises(ValueError):
        minimize(shifted_sphere, (1.0, -1.0), 3, pop_size=10, max_fes=100)


def test_paper_parameter_values_are_the_defaults():
    """Table 2: H = 0.5, G1 = 0.2, G2 = 0.1, alpha = 10, beta = 0.4."""
    config = get_config("edeco")
    assert (config.learning_habit, config.primary_school_ratio, config.middle_school_ratio) == (
        0.5,
        0.2,
        0.1,
    )
    assert (config.dfs_alpha, config.dfs_beta) == (10.0, 0.4)
    assert config.use_eda and config.use_dfs
    assert not get_config("ECO").use_eda and not get_config("ECO").use_dfs
    assert get_config("EECO").use_eda and not get_config("EECO").use_dfs
    assert get_config("DECO").use_dfs and not get_config("DECO").use_eda


# -------------------------------------------------------------------------- convergence


@pytest.mark.parametrize("algorithm", sorted(ALGORITHMS))
def test_convergence_on_the_sphere(algorithm):
    problem = get_problem("classical", "sphere", 10)
    result = minimize(
        problem.objective,
        problem.bounds,
        10,
        pop_size=30,
        max_fes=6000,
        seed=0,
        config=get_config(algorithm),
    )
    assert result.best_f < 1e-8


@pytest.mark.parametrize("algorithm", ["ECO", "EDECO"])
def test_convergence_on_a_shifted_sphere(algorithm):
    """The optimum is far from the origin; the initial values are of the order 1e4."""
    problem = get_problem("classical", "shifted_sphere", 10)
    result = minimize(
        problem.objective,
        problem.bounds,
        10,
        pop_size=50,
        max_fes=10_000,
        seed=0,
        config=get_config(algorithm),
    )
    assert result.history[0] > 1e3
    assert result.best_f < 50.0


def test_edeco_is_not_worse_than_eco_on_a_fixed_benchmark():
    """Equal budgets and fixed seeds on shifted unimodal and multimodal functions.

    The comparison is made on the mean of the log10 final values over all functions and
    seeds, which is insensitive to a single unlucky run.
    """
    functions = ["shifted_sphere", "shifted_ackley", "shifted_griewank"]
    seeds = range(6)
    log_values = {"ECO": [], "EDECO": []}
    wins = 0
    for name in functions:
        problem = get_problem("classical", name, 10)
        means = {}
        for algorithm in log_values:
            finals = [
                minimize(
                    problem.objective,
                    problem.bounds,
                    10,
                    pop_size=100,
                    max_fes=8000,
                    seed=seed,
                    config=get_config(algorithm),
                ).best_f
                for seed in seeds
            ]
            log_values[algorithm].extend(np.log10(np.maximum(finals, 1e-300)))
            means[algorithm] = float(np.mean(finals))
        wins += means["EDECO"] <= means["ECO"]
    assert np.mean(log_values["EDECO"]) <= np.mean(log_values["ECO"])
    assert wins >= 2
