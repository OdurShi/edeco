"""Dynamic fitness-distance balancing strategy (DFS), Section 3.2 of the paper.

DFS scores every individual by a weighted sum of its normalised fitness and its
normalised distance to the current best (Eq. 16-17).  The weight changes over the run
(Eq. 18).  The individual with the highest score, ``X_DFS``, replaces the leading term
of the high-school student update (Eq. 19).
"""

from __future__ import annotations

import math

import numpy as np

WEIGHT_SCHEDULES = ("sawtooth", "literal")


def dfs_weight(
    t: int, t_max: int, alpha: float = 10.0, beta: float = 0.4, schedule: str = "sawtooth"
) -> float:
    """Fitness weight ``omega`` of Eq. (18).

    ``schedule="sawtooth"`` (default) implements the dynamic FDB weight of Kahraman et
    al. that the equation is derived from: the run is divided into ``alpha`` periods of
    ``round(t_max / alpha)`` iterations and, within each period, the weight grows linearly
    from ``beta`` towards 1,

        omega = mod(t, t_max / alpha) / (t_max / alpha) * (1 - beta) + beta.

    ``schedule="literal"`` evaluates Eq. (18) exactly as printed,

        omega = mod(t_max / alpha, t) / t_max * (1 - beta) + beta,

    which stays within ``[beta, beta + (1 - beta) / alpha]`` (0.40 to 0.46 with the
    paper's parameters) and is therefore essentially constant.  See
    ``docs/implementation_notes.md`` for the reasoning behind the default.
    """
    if schedule == "sawtooth":
        period = max(1, math.floor(t_max / alpha + 0.5))
        fraction = (t % period) / period
    elif schedule == "literal":
        fraction = math.fmod(t_max / alpha, t) / t_max
    else:
        raise ValueError(f"unknown DFS weight schedule {schedule!r}; use {WEIGHT_SCHEDULES}")
    return fraction * (1.0 - beta) + beta


def _min_max(values: np.ndarray) -> np.ndarray:
    """Min-max normalisation to ``[0, 1]``; a constant vector maps to zeros."""
    low = values.min()
    span = values.max() - low
    if not np.isfinite(span) or span <= 0.0:
        return np.zeros_like(values, dtype=float)
    return (values - low) / span


def dfs_scores(population: np.ndarray, fitness: np.ndarray, weight: float) -> np.ndarray:
    """DFS score ``S_i = omega * norm(Fit_i) + (1 - omega) * norm(Dis_i)``, Eq. (16).

    ``Dis_i`` is the Euclidean distance to the best individual of the population
    (Eq. 17).  For a minimisation problem the normalised fitness is oriented so that the
    best individual scores 1 and the worst 0, the convention of the fitness-distance
    balance literature; the normalised distance is 0 for the best individual and 1 for
    the farthest one.  Infinite fitness values are treated as the worst finite value.
    """
    fitness = np.asarray(fitness, dtype=float)
    finite = np.isfinite(fitness)
    if not finite.all():
        worst = fitness[finite].max() if finite.any() else 0.0
        fitness = np.where(finite, fitness, worst)
    best = population[int(np.argmin(fitness))]
    distance = np.sqrt(((population - best) ** 2).sum(axis=1))
    norm_fitness = 1.0 - _min_max(fitness)
    norm_distance = _min_max(distance)
    return weight * norm_fitness + (1.0 - weight) * norm_distance


def select_dfs(population: np.ndarray, fitness: np.ndarray, weight: float) -> int:
    """Index of ``X_DFS``, the individual with the highest DFS score (first on ties)."""
    return int(np.argmax(dfs_scores(population, fitness, weight)))
