"""Benchmark problems: classical functions, CEC 2017 and engineering design problems."""

from __future__ import annotations

import numpy as np

from edeco.benchmarks import cec2017, classical, engineering
from edeco.benchmarks.problem import Problem

SUITES = ("classical", "cec2017", "engineering")


def get_problem(
    suite: str, function: str | int, dim: int | None = None, **options: float
) -> Problem:
    """Build a benchmark problem.

    Parameters
    ----------
    suite:
        ``"classical"``, ``"cec2017"`` or ``"engineering"``.
    function:
        Function name (classical), official function number (CEC 2017: 1, 3, ..., 30) or
        problem key (engineering: ``"RW01"`` ... ``"RW10"``).
    dim:
        Dimension; required for the scalable suites, ignored for engineering problems.
    options:
        ``penalty`` for the engineering suite (penalty coefficient).
    """
    if suite == "classical":
        if dim is None:
            raise ValueError("dim is required for the classical suite")
        return classical.make_problem(str(function), dim)
    if suite == "cec2017":
        if dim is None:
            raise ValueError("dim is required for the cec2017 suite")
        func = cec2017.make_function(int(function), dim)
        return Problem(
            name=f"F{int(function)}",
            suite="cec2017",
            dim=dim,
            lower=np.full(dim, cec2017.LOWER_BOUND),
            upper=np.full(dim, cec2017.UPPER_BOUND),
            objective=func,
            f_opt=func.f_opt,
        )
    if suite == "engineering":
        penalty = float(options.get("penalty", engineering.DEFAULT_PENALTY))
        return engineering.get_engineering_problem(str(function)).to_problem(penalty)
    raise KeyError(f"unknown suite {suite!r}; available: {SUITES}")


def list_functions(suite: str) -> list[str | int]:
    """Function identifiers of a suite, in the order used by the paper's tables."""
    if suite == "classical":
        return list(classical.FUNCTIONS)
    if suite == "cec2017":
        return list(cec2017.FUNCTION_IDS)
    if suite == "engineering":
        return list(engineering.PROBLEMS)
    raise KeyError(f"unknown suite {suite!r}; available: {SUITES}")


__all__ = ["SUITES", "Problem", "get_problem", "list_functions"]
