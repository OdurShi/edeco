"""Classical scalable test functions (built in, no external data needed).

These functions are not part of the paper's experiments.  They provide a quick,
dependency-free playground for the examples and the test-suite.  Every function has its
global minimum value 0; the ``shifted_*`` variants move the optimum away from the origin,
which matters for ECO because several of its update rules scale positions instead of
adding differences.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from edeco.benchmarks.problem import Problem


def sphere(x: np.ndarray) -> float:
    """Sphere function ``sum x_i^2``."""
    return float(np.sum(x * x))


def schwefel_222(x: np.ndarray) -> float:
    """Schwefel's problem 2.22 ``sum |x_i| + prod |x_i|``."""
    a = np.abs(x)
    return float(np.sum(a) + np.prod(a))


def schwefel_12(x: np.ndarray) -> float:
    """Schwefel's problem 1.2 (rotated hyper-ellipsoid)."""
    return float(np.sum(np.cumsum(x) ** 2))


def rosenbrock(x: np.ndarray) -> float:
    """Rosenbrock's valley; optimum at ``(1, ..., 1)``."""
    return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1.0) ** 2))


def step(x: np.ndarray) -> float:
    """Step function ``sum floor(x_i + 0.5)^2``."""
    return float(np.sum(np.floor(x + 0.5) ** 2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin's function."""
    return float(np.sum(x * x - 10.0 * np.cos(2.0 * math.pi * x) + 10.0))


def ackley(x: np.ndarray) -> float:
    """Ackley's function."""
    n = x.size
    return float(
        -20.0 * math.exp(-0.2 * math.sqrt(float(np.sum(x * x)) / n))
        - math.exp(float(np.sum(np.cos(2.0 * math.pi * x))) / n)
        + 20.0
        + math.e
    )


def griewank(x: np.ndarray) -> float:
    """Griewank's function."""
    i = np.arange(1, x.size + 1)
    return float(np.sum(x * x) / 4000.0 - np.prod(np.cos(x / np.sqrt(i))) + 1.0)


def zakharov(x: np.ndarray) -> float:
    """Zakharov's function."""
    s = float(np.sum(0.5 * np.arange(1, x.size + 1) * x))
    return float(np.sum(x * x)) + s**2 + s**4


def _shifted(func: Callable[[np.ndarray], float], offset: float) -> Callable[[np.ndarray], float]:
    def shifted(x: np.ndarray) -> float:
        return func(x - offset)

    return shifted


# name -> (function, lower bound, upper bound)
FUNCTIONS: dict[str, tuple[Callable[[np.ndarray], float], float, float]] = {
    "sphere": (sphere, -100.0, 100.0),
    "schwefel_222": (schwefel_222, -10.0, 10.0),
    "schwefel_12": (schwefel_12, -100.0, 100.0),
    "rosenbrock": (rosenbrock, -30.0, 30.0),
    "step": (step, -100.0, 100.0),
    "rastrigin": (rastrigin, -5.12, 5.12),
    "ackley": (ackley, -32.0, 32.0),
    "griewank": (griewank, -600.0, 600.0),
    "zakharov": (zakharov, -5.0, 10.0),
    "shifted_sphere": (_shifted(sphere, 30.0), -100.0, 100.0),
    "shifted_rastrigin": (_shifted(rastrigin, 1.5), -5.12, 5.12),
    "shifted_ackley": (_shifted(ackley, 10.0), -32.0, 32.0),
    "shifted_griewank": (_shifted(griewank, 150.0), -600.0, 600.0),
}


def make_problem(name: str, dim: int) -> Problem:
    """Build the classical function ``name`` in ``dim`` dimensions."""
    if name not in FUNCTIONS:
        raise KeyError(f"unknown classical function {name!r}; available: {sorted(FUNCTIONS)}")
    if dim < 2:
        raise ValueError("classical functions need at least two dimensions")
    func, lower, upper = FUNCTIONS[name]
    return Problem(
        name=name,
        suite="classical",
        dim=dim,
        lower=np.full(dim, lower),
        upper=np.full(dim, upper),
        objective=func,
        f_opt=0.0,
    )
