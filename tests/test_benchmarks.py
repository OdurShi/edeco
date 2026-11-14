"""Tests of the classical functions, the CEC 2017 suite and the engineering problems."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from edeco.benchmarks import cec2017, classical, engineering, get_problem, list_functions

# ----------------------------------------------------------------------------- classical

OPTIMA = {
    "sphere": 0.0,
    "schwefel_222": 0.0,
    "schwefel_12": 0.0,
    "rosenbrock": 1.0,
    "step": 0.0,
    "rastrigin": 0.0,
    "ackley": 0.0,
    "griewank": 0.0,
    "zakharov": 0.0,
    "shifted_sphere": 30.0,
    "shifted_rastrigin": 1.5,
    "shifted_ackley": 10.0,
    "shifted_griewank": 150.0,
}


@pytest.mark.parametrize("name", sorted(classical.FUNCTIONS))
def test_classical_functions_have_their_minimum_where_expected(name):
    problem = get_problem("classical", name, 6)
    x_opt = np.full(6, OPTIMA[name])
    assert np.all(x_opt >= problem.lower) and np.all(x_opt <= problem.upper)
    assert problem.objective(x_opt) == pytest.approx(0.0, abs=1e-12)
    rng = np.random.default_rng(0)
    for _ in range(20):
        x = rng.uniform(problem.lower, problem.upper)
        assert problem.objective(x) >= 0.0
    assert problem.f_opt == 0.0 and problem.dim == 6


def test_registry_lists_every_suite():
    assert list_functions("classical") == list(classical.FUNCTIONS)
    assert list_functions("cec2017") == [1, *range(3, 31)]
    assert list_functions("engineering") == [f"RW{i:02d}" for i in range(1, 11)]
    with pytest.raises(KeyError):
        get_problem("cec2099", 1, 10)
    with pytest.raises(KeyError):
        get_problem("classical", "does_not_exist", 10)


# ------------------------------------------------------------------------------ CEC 2017

needs_cec_data = pytest.mark.skipif(
    not cec2017.is_available(), reason="CEC 2017 data files not installed (pip install edeco[cec])"
)
REFERENCE = json.loads(
    (Path(__file__).parent / "data" / "cec2017_reference.json").read_text(encoding="utf-8")
)["values"]


def test_cec2017_suite_definition():
    """29 functions: the official F1 and F3-F30 (F2 was deleted from the suite)."""
    assert len(cec2017.FUNCTION_IDS) == 29 and 2 not in cec2017.FUNCTION_IDS
    assert cec2017.SUPPORTED_DIMS == (10, 30, 50, 100)
    assert (cec2017.LOWER_BOUND, cec2017.UPPER_BOUND) == (-100.0, 100.0)


@needs_cec_data
@pytest.mark.parametrize("dim", cec2017.SUPPORTED_DIMS)
def test_cec2017_matches_the_official_c_implementation(dim):
    """Probe points evaluated with the organisers' C code (see the reference file)."""
    for func_num in cec2017.FUNCTION_IDS:
        function = cec2017.make_function(func_num, dim)
        points = cec2017.validation_points(func_num, dim, function.x_opt)
        ours = np.array([function(point) for point in points])
        np.testing.assert_allclose(
            ours, REFERENCE[str(dim)][str(func_num)], rtol=1e-9, err_msg=f"F{func_num} D{dim}"
        )


@needs_cec_data
@pytest.mark.parametrize("dim", [10, 30])
def test_cec2017_optimum_values(dim):
    """Table 1 of the paper: F_i* = 100 * i, attained at the stored optimum location."""
    for func_num in cec2017.FUNCTION_IDS:
        function = cec2017.make_function(func_num, dim)
        assert function.f_opt == 100.0 * func_num
        assert np.all(np.abs(function.x_opt) <= 100.0)
        assert function(function.x_opt) == pytest.approx(function.f_opt, abs=1e-6)
        nearby = function.x_opt + 0.5
        assert function(nearby) > function.f_opt


@needs_cec_data
def test_cec2017_argument_validation():
    with pytest.raises(ValueError):
        cec2017.CEC2017Function(2, 10)
    with pytest.raises(ValueError):
        cec2017.CEC2017Function(5, 20)
    with pytest.raises(ValueError):
        cec2017.make_function(5, 10)(np.zeros(11))
    problem = get_problem("cec2017", 7, 10)
    assert problem.name == "F7" and problem.f_opt == 700.0 and problem.dim == 10


def test_cec2017_reports_missing_data(tmp_path):
    with pytest.raises(cec2017.CEC2017DataError):
        cec2017.find_data_dir(tmp_path)
    assert not cec2017.is_available(tmp_path)


# --------------------------------------------------------------------------- engineering

# Dimensions of Table 7 of the paper.
DIMENSIONS = {
    "RW01": 3, "RW02": 4, "RW03": 2, "RW04": 4, "RW05": 7,
    "RW06": 4, "RW07": 10, "RW08": 5, "RW09": 5, "RW10": 5,
}  # fmt: skip


@pytest.mark.parametrize("key", sorted(engineering.PROBLEMS))
def test_reference_solution_is_feasible_and_has_the_published_value(key):
    design = engineering.get_engineering_problem(key)
    assert design.dim == DIMENSIONS[key] == design.reference_x.size
    x = design.reference_x
    assert np.all(x >= design.lower) and np.all(x <= design.upper)
    assert design.max_violation(x) <= engineering.FEASIBILITY_TOL
    assert design.is_feasible(x)
    assert design.objective(x) == pytest.approx(design.reference_f, rel=2e-6)
    # A feasible design is not penalised.
    assert design.penalized(x) == pytest.approx(design.objective(x), rel=1e-9, abs=1e-18)


@pytest.mark.parametrize("key", sorted(engineering.PROBLEMS))
def test_penalised_problem_wrapper(key):
    design = engineering.get_engineering_problem(key)
    problem = get_problem("engineering", key)
    assert problem.suite == "engineering" and problem.dim == design.dim
    assert problem.objective(design.reference_x) == pytest.approx(
        design.penalized(design.reference_x)
    )
    weak = get_problem("engineering", key, penalty=1.0)
    rng = np.random.default_rng(1)
    for _ in range(50):
        x = rng.uniform(design.lower, design.upper)
        strong_value, weak_value = problem.objective(x), weak.objective(x)
        assert not np.isnan(strong_value)
        if design.max_violation(x) > 0.0 and np.isfinite(strong_value):
            assert strong_value > design.objective(x)
            assert strong_value >= weak_value


@pytest.mark.parametrize(
    ("key", "infeasible"),
    [
        ("RW01", [0.05, 0.25, 2.0]),  # very soft spring: deflection constraint violated
        ("RW02", [0.1, 0.1, 40.0, 200.0]),  # shell thinner than 0.0193 R
        ("RW03", [0.05, 0.05]),  # cross-sections too small for the load
        ("RW04", [0.1, 0.1, 0.1, 0.1]),  # shear stress far above the limit
        ("RW05", [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]),  # bending stress constraints
        ("RW07", [125.0, 21.0, 50.0, 0.515, 0.515, 0.4, 0.7, 0.3, 0.05, 0.8]),  # 50 balls
        ("RW08", [1.0, 1.0, 1.0, 1.0, 1.0]),  # deflection constraint
        ("RW09", [80.0, 90.0, 1.0, 600.0, 2.0]),  # r_o - r_i < 20 mm
        ("RW10", [10.0, 20.0, 30.0, 40.0, 50.0]),  # belt-length equalities
    ],
)
def test_infeasible_designs_are_detected_and_penalised(key, infeasible):
    design = engineering.get_engineering_problem(key)
    x = np.array(infeasible, dtype=float)
    assert design.max_violation(x) > engineering.FEASIBILITY_TOL
    assert not design.is_feasible(x)
    assert design.penalized(x) > design.objective(x) + 1.0


def test_discrete_variables_are_repaired():
    gear = engineering.get_engineering_problem("RW06")
    assert gear.objective(np.array([49.3, 15.8, 19.4, 42.6])) == pytest.approx(
        2.700857e-12, rel=1e-6
    )
    clutch = engineering.get_engineering_problem("RW09")
    np.testing.assert_array_equal(
        clutch.repair(np.array([69.6, 90.4, 1.2, 996.0, 2.7])), [70.0, 90.0, 1.0, 1000.0, 3.0]
    )
    bearing = engineering.get_engineering_problem("RW07")
    assert bearing.repair(bearing.reference_x + np.eye(10)[2] * 0.3)[2] == 11.0


def test_best_known_values_cannot_be_improved_by_nearby_feasible_designs():
    """Local optimality check of the reference solutions by random feasible perturbation."""
    rng = np.random.default_rng(3)
    for key, design in engineering.PROBLEMS.items():
        if key in ("RW06", "RW09"):  # purely discrete problems
            continue
        span = design.upper - design.lower
        for _ in range(300):
            x = np.clip(
                design.reference_x + 1e-3 * span * rng.standard_normal(design.dim),
                design.lower,
                design.upper,
            )
            if design.is_feasible(x, tol=0.0):
                tolerance = 2e-6 * abs(design.reference_f) + 1e-12
                assert design.objective(x) >= design.reference_f - tolerance, key
