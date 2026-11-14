"""Estimation-of-distribution step of the hybrid search framework (Section 3.1).

After the ECO sweep of an iteration, the fittest individuals (the *dominant group*) are
used to fit a multivariate Gaussian (Eqs. 12-14) from which new individuals are sampled
(Eq. 15).  The union of the ECO population and the samples is then truncated back to the
population size by fitness (the "greedy strategy" of Fig. 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GaussianModel:
    """Gaussian probability model ``G(X)_(mu, C)`` of Eq. (12).

    The model is stored as the mean and the centred dominant individuals rather than as
    an explicit covariance matrix.  Sampling ``mu + A^T z / sqrt(N_d)`` with
    ``z ~ N(0, I_{N_d})`` has covariance ``A^T A / N_d = C`` exactly, needs no matrix
    factorisation and remains valid when ``C`` is singular, which is always the case when
    the dominant group is not larger than the problem dimension.
    """

    mean: np.ndarray
    deviations: np.ndarray

    @property
    def n_dominant(self) -> int:
        """Number of dominant individuals ``N_d`` the model was fitted on."""
        return int(self.deviations.shape[0])

    @property
    def covariance(self) -> np.ndarray:
        """Covariance matrix ``C = (1 / N_d) * sum (X_i - mu)(X_i - mu)^T``, Eq. (14)."""
        return self.deviations.T @ self.deviations / self.n_dominant

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` individuals ``X = mu + g`` with ``g ~ N(0, C)``, Eq. (15)."""
        z = rng.standard_normal((n, self.n_dominant))
        return self.mean + z @ self.deviations / math.sqrt(self.n_dominant)


def select_dominant(fitness: np.ndarray, n_dominant: int) -> np.ndarray:
    """Indices of the ``n_dominant`` fittest individuals (ties keep the lower index)."""
    order = np.argsort(fitness, kind="stable")
    return order[:n_dominant]


def fit_gaussian(dominant: np.ndarray) -> GaussianModel:
    """Fit the Gaussian model to the dominant group.

    The mean is the unweighted average of Eq. (13) and the covariance uses the ``1/N_d``
    normalisation of Eq. (14).
    """
    if dominant.ndim != 2 or dominant.shape[0] < 1:
        raise ValueError("dominant group must be a non-empty (N_d, D) array")
    mean = dominant.mean(axis=0)
    return GaussianModel(mean=mean, deviations=dominant - mean)


def truncate_union(
    population: np.ndarray,
    fitness: np.ndarray,
    samples: np.ndarray,
    sample_fitness: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Greedy selection of the next population from ECO individuals and EDA samples.

    The two groups are pooled and the ``N`` fittest are kept, sorted by fitness.  Ties are
    broken in favour of the incumbent ECO individuals (stable sort with ECO first).

    Returns
    -------
    ``(population, fitness, n_survivors)`` where ``n_survivors`` counts the EDA samples
    that entered the next population.
    """
    n = population.shape[0]
    pooled_x = np.vstack([population, samples])
    pooled_f = np.concatenate([fitness, sample_fitness])
    order = np.argsort(pooled_f, kind="stable")[:n]
    return pooled_x[order], pooled_f[order], int(np.count_nonzero(order >= n))


def pairwise_replace(
    population: np.ndarray,
    fitness: np.ndarray,
    samples: np.ndarray,
    sample_fitness: np.ndarray,
    targets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Alternative greedy selection: each EDA sample competes with one ECO individual.

    Sample ``k`` replaces individual ``targets[k]`` when it is at least as fit.  The
    result is sorted by fitness.

    Returns
    -------
    ``(population, fitness, n_survivors)`` as in :func:`truncate_union`.
    """
    population = population.copy()
    fitness = fitness.copy()
    better = sample_fitness <= fitness[targets]
    population[targets[better]] = samples[better]
    fitness[targets[better]] = sample_fitness[better]
    order = np.argsort(fitness, kind="stable")
    return population[order], fitness[order], int(np.count_nonzero(better))
