"""Equation-level operators of the Educational Competition Optimizer (ECO).

Every function implements one numbered equation of the EDECO paper (Section 2) and is
free of side effects other than drawing from the supplied random generator, so each can
be unit-tested in isolation.  Where the paper's restatement of ECO differs from the
reference MATLAB implementation of ECO (Lian et al., 2024), both forms are provided and
the difference is described in ``docs/implementation_notes.md``.

Notation: ``tau = t / t_max`` is the elapsed fraction of the run, ``w`` is the step
weight of Eq. (3), ``P`` the random amplitude of Eq. (8) and ``c = P * E`` the product of
``P`` with the talent coefficient ``E`` of Eq. (9).
"""

from __future__ import annotations

import math

import numpy as np


def clamp(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Boundary control of ECO: coordinates outside ``[lb, ub]`` are set to the bound."""
    return np.minimum(np.maximum(x, lb), ub)


def logistic_map_init(
    rng: np.random.Generator, n: int, lb: np.ndarray, ub: np.ndarray
) -> np.ndarray:
    """Logistic-chaos population initialisation, Eqs. (1)-(2).

    Follows the reference ECO code: every coordinate starts from an independent uniform
    seed ``x0`` and applies one logistic-map step ``x = 4 * x0 * (1 - x0)`` before being
    scaled to the search range, ``X = lb + (ub - lb) * x``.
    """
    x0 = rng.random((n, lb.shape[0]))
    chaotic = 4.0 * x0 * (1.0 - x0)
    return clamp(lb + (ub - lb) * chaotic, lb, ub)


def levy_step(rng: np.random.Generator, dim: int, beta: float = 1.5) -> np.ndarray:
    """Lévy-distributed step ``Levy(D)`` of Eqs. (4) and (6) (Mantegna's algorithm)."""
    sigma = (
        math.gamma(1.0 + beta)
        * math.sin(math.pi * beta / 2.0)
        / (math.gamma((1.0 + beta) / 2.0) * beta * 2.0 ** ((beta - 1.0) / 2.0))
    ) ** (1.0 / beta)
    u = rng.standard_normal(dim) * sigma
    v = rng.standard_normal(dim)
    return u / np.abs(v) ** (1.0 / beta)


def step_weight(tau: float) -> float:
    """Adaptive step weight ``w = 0.1 * ln(2 - t / t_max)``, Eq. (3)."""
    return 0.1 * math.log(2.0 - tau)


def amplitude(tau: float, standard_normal: float) -> float:
    """Random amplitude ``P = 4 * randn * (1 - t / t_max)``, Eq. (8)."""
    return 4.0 * standard_normal * (1.0 - tau)


def talent_product(tau: float, p: float, r: float, h: float) -> float:
    """Product ``c = P * E`` with the talent coefficient ``E`` of Eq. (9).

    ``E = (pi / P) * (t / t_max)`` when ``R_t > H`` and ``E = 1`` otherwise, hence
    ``P * E = pi * t / t_max`` or ``P``.  Working with the product avoids the division by
    ``P`` (which is exactly zero in the last iteration) without changing any update rule,
    because ``E`` only ever appears multiplied by ``P``.
    """
    return math.pi * tau if r > h else p


def nearest_school(students: np.ndarray, schools: np.ndarray) -> np.ndarray:
    """``close(X_i)``: for every student the position of the nearest school.

    The reference ECO code measures proximity with the Manhattan (L1) distance and keeps
    the lowest-index school in the event of a tie; both conventions are reproduced.

    Parameters
    ----------
    students:
        Array of shape ``(m, D)``.
    schools:
        Array of shape ``(g, D)`` (the ``g`` best individuals of the population).

    Returns
    -------
    Array of shape ``(m, D)`` holding, for each student, the closest school.
    """
    distance = np.abs(students[:, None, :] - schools[None, :, :]).sum(axis=2)
    return schools[np.argmin(distance, axis=1)]


# --------------------------------------------------------------------------- stage 1


def primary_school_update(x: np.ndarray, w: float, levy: np.ndarray) -> np.ndarray:
    """Primary-school rule for schools, Eq. (4).

    ``X_imean`` is the mean of the agent's own coordinates (a scalar), as in the paper's
    definition "average position of the vector for the i-th school agent".
    """
    return x + w * (x.mean() - x) * levy


def primary_student_update(
    x: np.ndarray, w: float, close: np.ndarray, standard_normal: float
) -> np.ndarray:
    """Primary-school rule for students, Eq. (5)."""
    return x + w * (close - x) * standard_normal


# --------------------------------------------------------------------------- stage 2


def middle_school_update(
    x: np.ndarray, best: np.ndarray, population_mean: np.ndarray, tau: float, levy: np.ndarray
) -> np.ndarray:
    """Middle-school rule for schools, Eq. (6)."""
    return x + (best - population_mean) * math.exp(tau - 1.0) * levy


def middle_student_update(
    x: np.ndarray, w: float, close: np.ndarray, p: float, c: float
) -> np.ndarray:
    """Middle-school rule for students, Eq. (7).

    ``X - w * close - P * (E * w * close - X)`` written with ``c = P * E``.
    """
    return x - w * close - (c * w * close - p * x)


# --------------------------------------------------------------------------- stage 3


def high_school_update(
    x: np.ndarray, best: np.ndarray, anchor: np.ndarray, r1: float, r2: float
) -> np.ndarray:
    """High-school rule for schools, Eq. (10).

    ``X + (X_best - A) * randn - (X_best - A) * randn`` with two independent normal
    draws.  The reference ECO code uses the agent itself as anchor ``A = X_i``; the
    EDECO paper prints the population mean ``A = X_mean``.  The caller chooses.
    """
    return x + (best - anchor) * r1 - (best - anchor) * r2


def high_student_update(
    x: np.ndarray, best: np.ndarray, leader: np.ndarray, p: float, c: float
) -> np.ndarray:
    """High-school rule for students, Eq. (11), and its DFS form.

    ``L - P * (E * X_best - X)`` with ``c = P * E``.  The leading term ``L`` is the
    global best in the reference ECO code, ``X_i`` in Eq. (11) as printed in the EDECO
    paper, and ``X_DFS`` when the dynamic fitness-distance balancing strategy is active
    (default reading of Eq. 19); the caller chooses.
    """
    return leader - (c * best - p * x)


def dfs_student_update(
    x: np.ndarray, x_dfs: np.ndarray, w: float, close: np.ndarray, p: float, c: float
) -> np.ndarray:
    """Student rule of Eq. (19) exactly as printed.

    ``X_DFS - w * close(X) - P * (E * w * close(X) - X)``: Eq. (7) with the leading
    term replaced by the individual selected by the dynamic fitness-distance balance.
    The default EDECO configuration instead applies :func:`high_student_update` with
    ``X_DFS`` as leader (``dfs_student_rule="eq11"``); this function implements the
    alternative ``dfs_student_rule="eq19"``.
    """
    return x_dfs - w * close - (c * w * close - p * x)
