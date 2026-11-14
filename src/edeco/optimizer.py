"""ECO, EDECO and the ablation variants EECO / DECO behind one configuration object.

``ECOConfig(use_eda=False, use_dfs=False)`` is the base Educational Competition
Optimizer (Algorithm 1 of the paper, semantics of the reference ECO code).  Switching on
``use_eda`` adds the hybrid search framework with the estimation-of-distribution step
(Section 3.1) and ``use_dfs`` adds the dynamic fitness-distance balancing strategy
(Section 3.2).  Both together give EDECO (Algorithm 2).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import numpy as np

from edeco import operators as ops
from edeco.dfs import WEIGHT_SCHEDULES, dfs_weight, select_dfs
from edeco.eda import fit_gaussian, pairwise_replace, select_dominant, truncate_union
from edeco.evaluator import Evaluator, Objective

logger = logging.getLogger(__name__)

BASE_EQUATIONS = ("reference", "paper")
DFS_STUDENT_RULES = ("eq11", "eq19")
EDA_SELECTIONS = ("truncation", "pairwise")


def _round_half_up(value: float) -> int:
    """Round like MATLAB's ``round`` for non-negative values (half away from zero)."""
    return math.floor(value + 0.5)


@dataclass(frozen=True)
class ECOConfig:
    """Hyper-parameters of ECO / EDECO (Table 2 of the paper) and documented choices.

    Attributes
    ----------
    learning_habit:
        ``H`` of Eq. (9), 0.5 in the paper.
    primary_school_ratio, middle_school_ratio:
        ``G1 = 0.2`` and ``G2 = 0.1``: fraction of the population acting as schools in
        the primary stage and in the middle/high stages.
    levy_beta:
        Stability index of the Lévy step (1.5 in the reference ECO code).
    use_eda:
        Enable the hybrid search framework (HSF) with the EDA step, Eqs. (12)-(15).
    use_dfs:
        Enable the dynamic fitness-distance balancing strategy, Eqs. (16)-(19).
    dominant_ratio:
        ``N_d / N``, size of the dominant group.  Not stated in the paper; 0.5 follows
        the proportions drawn in its Fig. 1.
    eda_sample_ratio:
        ``N_E / N``, number of EDA samples per iteration.  Not stated in the paper; 1/3
        follows the proportions drawn in its Fig. 1.
    eda_selection:
        ``"truncation"`` keeps the ``N`` fittest of the pooled ECO individuals and EDA
        samples ("combining population and using greedy strategy", Fig. 1);
        ``"pairwise"`` lets every sample compete with one randomly chosen individual.
    dfs_alpha, dfs_beta:
        ``alpha = 10`` and ``beta = 0.4`` of Eq. (18).
    dfs_weight_schedule:
        ``"sawtooth"`` (dynamic-FDB reading of Eq. 18) or ``"literal"``.
    dfs_student_rule:
        How ``X_DFS`` enters the high-school student update.  ``"eq11"`` (default) keeps
        the high-school rule of Eq. (11) and replaces its leading best individual by
        ``X_DFS``; ``"eq19"`` uses Eq. (19) exactly as printed (the middle-school rule of
        Eq. (7) with ``X_DFS`` as leading term).  The paper is not consistent on this
        point; see ``docs/implementation_notes.md``.
    base_equations:
        ``"reference"`` uses the reference ECO code where it differs from the paper's
        restatement of Eqs. (10)-(11); ``"paper"`` uses the equations as printed.
    """

    learning_habit: float = 0.5
    primary_school_ratio: float = 0.2
    middle_school_ratio: float = 0.1
    levy_beta: float = 1.5
    use_eda: bool = False
    use_dfs: bool = False
    dominant_ratio: float = 0.5
    eda_sample_ratio: float = 1.0 / 3.0
    eda_selection: str = "truncation"
    dfs_alpha: float = 10.0
    dfs_beta: float = 0.4
    dfs_weight_schedule: str = "sawtooth"
    dfs_student_rule: str = "eq11"
    base_equations: str = "reference"

    def __post_init__(self) -> None:
        if not 0.0 < self.primary_school_ratio < 1.0 or not 0.0 < self.middle_school_ratio < 1.0:
            raise ValueError("school ratios must lie in (0, 1)")
        if not 0.0 < self.dominant_ratio <= 1.0:
            raise ValueError("dominant_ratio must lie in (0, 1]")
        if self.eda_sample_ratio <= 0.0:
            raise ValueError("eda_sample_ratio must be positive")
        if self.dfs_alpha <= 0.0 or not 0.0 <= self.dfs_beta <= 1.0:
            raise ValueError("dfs_alpha must be positive and dfs_beta must lie in [0, 1]")
        if self.dfs_weight_schedule not in WEIGHT_SCHEDULES:
            raise ValueError(f"dfs_weight_schedule must be one of {WEIGHT_SCHEDULES}")
        if self.dfs_student_rule not in DFS_STUDENT_RULES:
            raise ValueError(f"dfs_student_rule must be one of {DFS_STUDENT_RULES}")
        if self.eda_selection not in EDA_SELECTIONS:
            raise ValueError(f"eda_selection must be one of {EDA_SELECTIONS}")
        if self.eda_selection == "pairwise" and self.eda_sample_ratio > 1.0:
            raise ValueError("pairwise selection needs eda_sample_ratio <= 1")
        if self.base_equations not in BASE_EQUATIONS:
            raise ValueError(f"base_equations must be one of {BASE_EQUATIONS}")


#: The four algorithms evaluated in the paper's ablation study (Section 4.2).
ALGORITHMS: dict[str, ECOConfig] = {
    "ECO": ECOConfig(),
    "EECO": ECOConfig(use_eda=True),
    "DECO": ECOConfig(use_dfs=True),
    "EDECO": ECOConfig(use_eda=True, use_dfs=True),
}


def get_config(algorithm: str, **overrides: object) -> ECOConfig:
    """Configuration of a named algorithm (case-insensitive) with optional overrides."""
    key = algorithm.upper()
    if key not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {algorithm!r}; available: {sorted(ALGORITHMS)}")
    return replace(ALGORITHMS[key], **overrides) if overrides else ALGORITHMS[key]


@dataclass
class OptimizeResult:
    """Outcome of one optimisation run.

    Attributes
    ----------
    best_x, best_f:
        Best solution found and its objective value.
    history:
        Best-so-far objective value after each evaluation (length ``n_evals``).
    n_evals:
        Number of objective evaluations actually performed (never above the budget).
    n_iterations:
        Number of ECO iterations started.
    info:
        Counters describing the run (EDA samples drawn / accepted, population size...).
    """

    best_x: np.ndarray
    best_f: float
    history: np.ndarray
    n_evals: int
    n_iterations: int
    info: dict[str, float] = field(default_factory=dict)


def _as_bounds(
    bounds: tuple[float | Sequence[float], float | Sequence[float]], dim: int
) -> tuple[np.ndarray, np.ndarray]:
    lb = np.broadcast_to(np.asarray(bounds[0], dtype=float), (dim,)).copy()
    ub = np.broadcast_to(np.asarray(bounds[1], dtype=float), (dim,)).copy()
    if not np.all(ub > lb):
        raise ValueError("every upper bound must be larger than its lower bound")
    return lb, ub


def minimize(
    objective: Objective,
    bounds: tuple[float | Sequence[float], float | Sequence[float]],
    dim: int,
    *,
    pop_size: int = 100,
    max_fes: int = 50_000,
    seed: int | np.random.SeedSequence | None = None,
    config: ECOConfig | None = None,
) -> OptimizeResult:
    """Minimise ``objective`` with ECO or one of its enhanced variants.

    Parameters
    ----------
    objective:
        Function of a 1-D array of length ``dim`` returning a scalar to minimise.
    bounds:
        ``(lower, upper)``; each a scalar or a sequence of length ``dim``.
    dim:
        Problem dimension ``D``.
    pop_size:
        Population size ``N`` (at least 4).  The paper does not state the value used in
        its experiments; 100 is the default of this package (see the implementation
        notes for the sensitivity of the results to this choice).
    max_fes:
        Evaluation budget ``Fes_max``; the run performs exactly this many evaluations.
    seed:
        Seed of the NumPy generator; runs with equal seeds are identical.
    config:
        Algorithm configuration; defaults to base ECO.  Use :func:`get_config`.

    Returns
    -------
    :class:`OptimizeResult`
    """
    cfg = config if config is not None else ECOConfig()
    if pop_size < 4:
        raise ValueError("pop_size must be at least 4")
    lb, ub = _as_bounds(bounds, dim)
    rng = np.random.default_rng(seed)
    evaluate = Evaluator(objective, max_fes)

    n = int(pop_size)
    g1 = max(1, _round_half_up(n * cfg.primary_school_ratio))
    g2 = max(1, _round_half_up(n * cfg.middle_school_ratio))
    n_dominant = max(2, _round_half_up(n * cfg.dominant_ratio)) if cfg.use_eda else 0
    n_eda = max(1, _round_half_up(n * cfg.eda_sample_ratio)) if cfg.use_eda else 0

    # Line 2 of Algorithms 1-2: logistic-map initialisation, then fitness and sorting.
    population = ops.logistic_map_init(rng, n, lb, ub)
    fitness = np.full(n, np.inf)
    for j in range(min(n, max_fes)):
        fitness[j] = evaluate(population[j])
    order = np.argsort(fitness, kind="stable")
    population, fitness = population[order], fitness[order]

    # The paper stops on evaluations; the iteration horizon t_max used by Eqs. (3), (6),
    # (8), (9) and (18) is the number of iterations that fit in the remaining budget.
    evals_per_iteration = n + n_eda
    t_max = max(1, math.ceil((max_fes - n) / evals_per_iteration))
    info = {"pop_size": n, "t_max": t_max, "eda_samples": 0, "eda_survivors": 0}

    t = 0
    while not evaluate.exhausted:
        t += 1
        tau = min(t / t_max, 1.0)
        w = ops.step_weight(tau)
        p = ops.amplitude(tau, rng.standard_normal())
        c = ops.talent_product(tau, p, rng.random(), cfg.learning_habit)
        stage = t % 3  # 1: primary school, 2: middle school, 0: high school

        new_population = population.copy()
        new_fitness = fitness.copy()
        n_schools = g1 if stage == 1 else g2
        schools = population[:n_schools]
        needs_close = stage in (1, 2) or (cfg.use_dfs and cfg.dfs_student_rule == "eq19")
        close = ops.nearest_school(population[n_schools:], schools) if needs_close else None
        population_mean = population.mean(axis=0)
        x_dfs = None
        if stage == 0 and cfg.use_dfs:
            weight = dfs_weight(t, t_max, cfg.dfs_alpha, cfg.dfs_beta, cfg.dfs_weight_schedule)
            x_dfs = population[select_dfs(population, fitness, weight)]

        for j in range(n):
            if evaluate.exhausted:
                break
            x = population[j]
            best = evaluate.best_x  # global best, updated within the sweep as in ECO
            is_school = j < n_schools
            if stage == 1:
                if is_school:
                    candidate = ops.primary_school_update(
                        x, w, ops.levy_step(rng, dim, cfg.levy_beta)
                    )
                else:
                    candidate = ops.primary_student_update(
                        x, w, close[j - n_schools], rng.standard_normal()
                    )
            elif stage == 2:
                if is_school:
                    candidate = ops.middle_school_update(
                        x, best, population_mean, tau, ops.levy_step(rng, dim, cfg.levy_beta)
                    )
                else:
                    candidate = ops.middle_student_update(x, w, close[j - n_schools], p, c)
            elif is_school:
                anchor = x if cfg.base_equations == "reference" else population_mean
                candidate = ops.high_school_update(
                    x, best, anchor, rng.standard_normal(), rng.standard_normal()
                )
            elif x_dfs is not None and cfg.dfs_student_rule == "eq19":
                candidate = ops.dfs_student_update(x, x_dfs, w, close[j - n_schools], p, c)
            else:
                if x_dfs is not None:
                    leader = x_dfs
                else:
                    leader = best if cfg.base_equations == "reference" else x
                candidate = ops.high_student_update(x, best, leader, p, c)

            candidate = ops.clamp(candidate, lb, ub)
            value = evaluate(candidate)
            if value <= new_fitness[j]:  # greedy replacement of the reference ECO code
                new_population[j] = candidate
                new_fitness[j] = value

        population, fitness = new_population, new_fitness

        if cfg.use_eda and not evaluate.exhausted:
            # Lines 24-26 of Algorithm 2: dominant group -> Gaussian model -> sampling ->
            # greedy selection over the pooled ECO and EDA individuals.
            dominant = population[select_dominant(fitness, n_dominant)]
            samples = ops.clamp(fit_gaussian(dominant).sample(n_eda, rng), lb, ub)
            samples = samples[: evaluate.remaining]
            sample_fitness = np.array([evaluate(s) for s in samples])
            if cfg.eda_selection == "truncation":
                population, fitness, survivors = truncate_union(
                    population, fitness, samples, sample_fitness
                )
            else:
                targets = rng.permutation(n)[: len(samples)]
                population, fitness, survivors = pairwise_replace(
                    population, fitness, samples, sample_fitness, targets
                )
            info["eda_samples"] += len(samples)
            info["eda_survivors"] += survivors
        else:
            order = np.argsort(fitness, kind="stable")
            population, fitness = population[order], fitness[order]

    logger.debug("finished after %d iterations and %d evaluations", t, evaluate.n_evals)
    assert evaluate.best_x is not None
    return OptimizeResult(
        best_x=evaluate.best_x,
        best_f=evaluate.best_f,
        history=evaluate.history(),
        n_evals=evaluate.n_evals,
        n_iterations=t,
        info=info,
    )
