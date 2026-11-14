"""Strict function-evaluation accounting shared by every optimiser in this package.

The paper stops every algorithm after a fixed number of objective evaluations
(``Fes_max = 5000 * D``, Section 4.1).  :class:`Evaluator` is the single place where an
objective is called, so the budget can never be exceeded and the convergence history is
recorded identically for every algorithm.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

Objective = Callable[[np.ndarray], float]


class BudgetExhaustedError(RuntimeError):
    """Raised when an evaluation is requested after the budget has been spent."""


class Evaluator:
    """Wrap an objective with an evaluation budget, best-so-far tracking and a history.

    Parameters
    ----------
    objective:
        Function mapping a 1-D array of length ``dim`` to a scalar (minimisation).
        It must not modify its argument.
    max_fes:
        Maximum number of objective evaluations (``Fes_max`` in the paper).
    """

    def __init__(self, objective: Objective, max_fes: int) -> None:
        if max_fes < 1:
            raise ValueError("max_fes must be a positive integer")
        self._objective = objective
        self.max_fes = int(max_fes)
        self.n_evals = 0
        self.best_f = float("inf")
        self.best_x: np.ndarray | None = None
        self._history = np.empty(self.max_fes, dtype=float)

    @property
    def remaining(self) -> int:
        """Number of evaluations still available."""
        return self.max_fes - self.n_evals

    @property
    def exhausted(self) -> bool:
        """``True`` once the whole budget has been spent."""
        return self.n_evals >= self.max_fes

    def __call__(self, x: np.ndarray) -> float:
        """Evaluate ``x``, count the evaluation and update the best-so-far record.

        Non-finite ``NaN`` objective values are mapped to ``+inf`` so that comparisons in
        the optimisers stay well defined.
        """
        if self.n_evals >= self.max_fes:
            raise BudgetExhaustedError(f"evaluation budget of {self.max_fes} exhausted")
        value = float(self._objective(x))
        if np.isnan(value):
            value = float("inf")
        if value < self.best_f or self.best_x is None:
            self.best_f = value
            self.best_x = np.array(x, dtype=float, copy=True)
        self._history[self.n_evals] = self.best_f
        self.n_evals += 1
        return value

    def history(self) -> np.ndarray:
        """Best-so-far objective value after each evaluation (length ``n_evals``)."""
        return self._history[: self.n_evals].copy()
