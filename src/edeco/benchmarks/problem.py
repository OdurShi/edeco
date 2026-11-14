"""Common description of a box-constrained minimisation problem."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Problem:
    """A minimisation problem with box constraints.

    Attributes
    ----------
    name:
        Short identifier, e.g. ``"F5"``, ``"sphere"`` or ``"RW01"``.
    suite:
        Benchmark suite the problem belongs to.
    dim:
        Number of decision variables.
    lower, upper:
        Bound vectors of length ``dim``.
    objective:
        Function to minimise.  For the engineering problems this is the penalised
        objective.
    f_opt:
        Known optimum value, or ``None`` when it is not known exactly.
    """

    name: str
    suite: str
    dim: int
    lower: np.ndarray
    upper: np.ndarray
    objective: Callable[[np.ndarray], float]
    f_opt: float | None = None

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """``(lower, upper)`` in the form expected by :func:`edeco.minimize`."""
        return self.lower, self.upper
