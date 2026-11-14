"""Minimal use of the library API: minimise a shifted sphere function with EDECO.

Run with ``python examples/quickstart.py`` (about a second on a laptop CPU).
"""

from __future__ import annotations

import numpy as np

from edeco import get_config, minimize

SHIFT = 30.0


def shifted_sphere(x: np.ndarray) -> float:
    """Sphere function with its optimum moved to (30, ..., 30); minimum value 0."""
    return float(np.sum((x - SHIFT) ** 2))


def main() -> None:
    dim = 10
    result = minimize(
        shifted_sphere,
        bounds=(-100.0, 100.0),
        dim=dim,
        pop_size=100,
        max_fes=5000 * dim,  # the paper's budget: 5000 evaluations per dimension
        seed=0,
        config=get_config("EDECO"),
    )
    print(f"evaluations : {result.n_evals} in {result.n_iterations} iterations")
    print(f"best value  : {result.best_f:.6e}")
    print(f"best x      : {np.array2string(result.best_x, precision=4)}")
    checkpoints = [0, 999, 9999, result.n_evals - 1]
    history = ", ".join(f"{result.history[i]:.3e}" for i in checkpoints)
    print(f"best-so-far after 1, 1000, 10000 and {result.n_evals} evaluations: {history}")


if __name__ == "__main__":
    main()
