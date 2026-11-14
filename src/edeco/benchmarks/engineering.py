"""Ten constrained engineering design problems (Table 7 of the paper, RW01-RW10).

The paper lists the problems by name and dimension only and states that "the penalty
function method was used to transform the constrained problems into unconstrained
optimization problems" (Section 4.4).  The formulations below are therefore the standard
ones from the engineering-optimisation literature (the references are given per problem)
and the penalty is the static quadratic penalty

    F(x) = f(x) + rho * sum_i max(0, g_i(x))^2,

where every constraint is written as ``g_i(x) <= 0`` and an equality ``h(x) = 0`` is
relaxed to ``|h(x)| - 1e-4 <= 0``.  ``rho`` defaults to :data:`DEFAULT_PENALTY`, which is
large enough for the penalised optimum to be feasible within :data:`FEASIBILITY_TOL`.

Several best values in the paper's Table 8 cannot be obtained with these formulations
(they are lower than the best known feasible optimum or of a different magnitude); see
``docs/implementation_notes.md`` for the comparison problem by problem.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from edeco.benchmarks.problem import Problem

DEFAULT_PENALTY = 1.0e12
FEASIBILITY_TOL = 1.0e-6
EQUALITY_TOL = 1.0e-4


def _identity(x: np.ndarray) -> np.ndarray:
    return x


@dataclass(frozen=True)
class EngineeringProblem:
    """A constrained design problem ``min f(x)  s.t.  g(x) <= 0,  lower <= x <= upper``.

    Attributes
    ----------
    key, name:
        Identifier used in the paper (``RW01`` ... ``RW10``) and descriptive name.
    lower, upper:
        Bounds of the design variables.
    raw_objective:
        The design objective ``f`` (without penalty).
    constraints:
        Function returning the vector ``g(x)``; the design is feasible when all entries
        are non-positive.
    repair:
        Maps a continuous candidate to an admissible design (e.g. rounds integer
        variables).  It is applied before the objective and the constraints.
    reference_x, reference_f:
        A solution reported in the literature and its objective value, used by the tests.
    source:
        Literature reference of the formulation and of the reference solution.
    """

    key: str
    name: str
    lower: np.ndarray
    upper: np.ndarray
    raw_objective: Callable[[np.ndarray], float]
    constraints: Callable[[np.ndarray], np.ndarray]
    reference_x: np.ndarray
    reference_f: float
    source: str
    repair: Callable[[np.ndarray], np.ndarray] = _identity

    @property
    def dim(self) -> int:
        """Number of design variables."""
        return int(self.lower.size)

    def max_violation(self, x: np.ndarray) -> float:
        """Largest constraint violation ``max(0, max_i g_i(x))`` of the repaired design.

        A design at which a constraint cannot be evaluated (division by zero at a bound)
        counts as infinitely infeasible.
        """
        with np.errstate(all="ignore"):
            g = self.constraints(self.repair(np.asarray(x, dtype=float)))
        if g.size == 0:
            return 0.0
        worst = float(np.max(g))
        return float("inf") if math.isnan(worst) else max(0.0, worst)

    def is_feasible(self, x: np.ndarray, tol: float = FEASIBILITY_TOL) -> bool:
        """``True`` when ``x`` respects the bounds and every constraint within ``tol``."""
        x = np.asarray(x, dtype=float)
        in_bounds = bool(np.all(x >= self.lower - tol) and np.all(x <= self.upper + tol))
        return in_bounds and self.max_violation(x) <= tol

    def objective(self, x: np.ndarray) -> float:
        """Design objective ``f`` of the repaired design."""
        with np.errstate(all="ignore"):
            return float(self.raw_objective(self.repair(np.asarray(x, dtype=float))))

    def penalized(self, x: np.ndarray, penalty: float = DEFAULT_PENALTY) -> float:
        """Static quadratic penalty function ``f + penalty * sum max(0, g)^2``."""
        x = self.repair(np.asarray(x, dtype=float))
        with np.errstate(all="ignore"):
            violation = np.maximum(self.constraints(x), 0.0)
            value = float(self.raw_objective(x)) + penalty * float(np.sum(violation**2))
        return value if math.isfinite(value) else float("inf")

    def to_problem(self, penalty: float = DEFAULT_PENALTY) -> Problem:
        """Unconstrained :class:`Problem` minimising the penalised objective."""

        def objective(x: np.ndarray) -> float:
            return self.penalized(x, penalty)

        return Problem(
            name=self.key,
            suite="engineering",
            dim=self.dim,
            lower=self.lower.copy(),
            upper=self.upper.copy(),
            objective=objective,
            f_opt=None,
        )


def _arr(*values: float) -> np.ndarray:
    return np.array(values, dtype=float)


# ---------------------------------------------------------------- RW01 spring


def _spring_f(x: np.ndarray) -> float:
    d, coil, n = x
    return (n + 2.0) * coil * d**2


def _spring_g(x: np.ndarray) -> np.ndarray:
    d, coil, n = x
    return _arr(
        1.0 - coil**3 * n / (71785.0 * d**4),
        (4.0 * coil**2 - d * coil) / (12566.0 * (coil * d**3 - d**4)) + 1.0 / (5108.0 * d**2) - 1.0,
        1.0 - 140.45 * d / (coil**2 * n),
        (d + coil) / 1.5 - 1.0,
    )


# ------------------------------------------------------- RW02 pressure vessel


def _vessel_f(x: np.ndarray) -> float:
    ts, th, r, length = x
    return (
        0.6224 * ts * r * length + 1.7781 * th * r**2 + 3.1661 * ts**2 * length + 19.84 * ts**2 * r
    )


def _vessel_g(x: np.ndarray) -> np.ndarray:
    ts, th, r, length = x
    return _arr(
        -ts + 0.0193 * r,
        -th + 0.00954 * r,
        -math.pi * r**2 * length - 4.0 / 3.0 * math.pi * r**3 + 1296000.0,
        length - 240.0,
    )


# -------------------------------------------------------- RW03 three-bar truss


def _truss_f(x: np.ndarray) -> float:
    a1, a2 = x
    return (2.0 * math.sqrt(2.0) * a1 + a2) * 100.0


def _truss_g(x: np.ndarray) -> np.ndarray:
    a1, a2 = x
    load, stress = 2.0, 2.0
    root2 = math.sqrt(2.0)
    denom = root2 * a1**2 + 2.0 * a1 * a2
    return _arr(
        (root2 * a1 + a2) / denom * load - stress,
        a2 / denom * load - stress,
        1.0 / (root2 * a2 + a1) * load - stress,
    )


# ----------------------------------------------------------- RW04 welded beam

_WB_P, _WB_L, _WB_E, _WB_G = 6000.0, 14.0, 30.0e6, 12.0e6
_WB_TAU_MAX, _WB_SIGMA_MAX, _WB_DELTA_MAX = 13600.0, 30000.0, 0.25


def _beam_f(x: np.ndarray) -> float:
    h, length, t, b = x
    return 1.10471 * h**2 * length + 0.04811 * t * b * (14.0 + length)


def _beam_g(x: np.ndarray) -> np.ndarray:
    h, length, t, b = x
    tau1 = _WB_P / (math.sqrt(2.0) * h * length)
    moment = _WB_P * (_WB_L + length / 2.0)
    radius = math.sqrt(length**2 / 4.0 + ((h + t) / 2.0) ** 2)
    polar = 2.0 * (math.sqrt(2.0) * h * length * (length**2 / 12.0 + ((h + t) / 2.0) ** 2))
    tau2 = moment * radius / polar
    tau = math.sqrt(tau1**2 + 2.0 * tau1 * tau2 * length / (2.0 * radius) + tau2**2)
    sigma = 6.0 * _WB_P * _WB_L / (b * t**2)
    delta = 4.0 * _WB_P * _WB_L**3 / (_WB_E * t**3 * b)
    buckling = (
        4.013
        * _WB_E
        * math.sqrt(t**2 * b**6 / 36.0)
        / _WB_L**2
        * (1.0 - t / (2.0 * _WB_L) * math.sqrt(_WB_E / (4.0 * _WB_G)))
    )
    return _arr(
        tau - _WB_TAU_MAX,
        sigma - _WB_SIGMA_MAX,
        h - b,
        0.10471 * h**2 + 0.04811 * t * b * (14.0 + length) - 5.0,
        0.125 - h,
        delta - _WB_DELTA_MAX,
        _WB_P - buckling,
    )


# --------------------------------------------------------- RW05 speed reducer


def _reducer_f(x: np.ndarray) -> float:
    x1, x2, x3, x4, x5, x6, x7 = x
    return (
        0.7854 * x1 * x2**2 * (3.3333 * x3**2 + 14.9334 * x3 - 43.0934)
        - 1.508 * x1 * (x6**2 + x7**2)
        + 7.4777 * (x6**3 + x7**3)
        + 0.7854 * (x4 * x6**2 + x5 * x7**2)
    )


def _reducer_g(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4, x5, x6, x7 = x
    return _arr(
        27.0 / (x1 * x2**2 * x3) - 1.0,
        397.5 / (x1 * x2**2 * x3**2) - 1.0,
        1.93 * x4**3 / (x2 * x3 * x6**4) - 1.0,
        1.93 * x5**3 / (x2 * x3 * x7**4) - 1.0,
        math.sqrt((745.0 * x4 / (x2 * x3)) ** 2 + 16.9e6) / (110.0 * x6**3) - 1.0,
        math.sqrt((745.0 * x5 / (x2 * x3)) ** 2 + 157.5e6) / (85.0 * x7**3) - 1.0,
        x2 * x3 / 40.0 - 1.0,
        5.0 * x2 / x1 - 1.0,
        x1 / (12.0 * x2) - 1.0,
        (1.5 * x6 + 1.9) / x4 - 1.0,
        (1.1 * x7 + 1.9) / x5 - 1.0,
    )


def _reducer_repair(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[2] = np.round(x[2])  # number of pinion teeth is an integer
    return x


# ------------------------------------------------------------- RW06 gear train


def _gear_f(x: np.ndarray) -> float:
    ta, tb, td, tf = x
    return (1.0 / 6.931 - td * tb / (ta * tf)) ** 2


def _no_constraints(x: np.ndarray) -> np.ndarray:
    return np.empty(0)


# ------------------------------------------------- RW07 rolling element bearing

_RB_D, _RB_d, _RB_BW = 160.0, 90.0, 30.0


def _bearing_repair(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[2] = np.round(x[2])  # number of balls Z is an integer
    return x


def _bearing_f(x: np.ndarray) -> float:
    dm, db, z, fi, fo = x[:5]
    gamma = db / dm
    fc = (
        37.91
        * (
            1.0
            + (
                1.04
                * ((1.0 - gamma) / (1.0 + gamma)) ** 1.72
                * (fi * (2.0 * fo - 1.0) / (fo * (2.0 * fi - 1.0))) ** 0.41
            )
            ** (10.0 / 3.0)
        )
        ** (-0.3)
        * (gamma**0.3 * (1.0 - gamma) ** 1.39 / (1.0 + gamma) ** (1.0 / 3.0))
        * (2.0 * fi / (2.0 * fi - 1.0)) ** 0.41
    )
    if db <= 25.4:
        capacity = fc * z ** (2.0 / 3.0) * db**1.8
    else:
        capacity = 3.647 * fc * z ** (2.0 / 3.0) * db**1.4
    return -capacity  # the dynamic load capacity is maximised


def _bearing_g(x: np.ndarray) -> np.ndarray:
    dm, db, z, fi, fo, kd_min, kd_max, eps, e, zeta = x
    big_d, small_d, width = _RB_D, _RB_d, _RB_BW
    t = big_d - small_d - 2.0 * db
    a = (big_d - small_d) / 2.0 - 3.0 * t / 4.0
    b = big_d / 2.0 - t / 4.0 - db
    c = small_d / 2.0 + t / 4.0
    cosine = (a**2 + b**2 - c**2) / (2.0 * a * b)
    phi0 = 2.0 * math.pi - 2.0 * math.acos(min(1.0, max(-1.0, cosine)))
    return _arr(
        z - 1.0 - phi0 / (2.0 * math.asin(db / dm)),
        kd_min * (big_d - small_d) - 2.0 * db,
        2.0 * db - kd_max * (big_d - small_d),
        db - zeta * width,
        0.5 * (big_d + small_d) - dm,
        dm - (0.5 + e) * (big_d + small_d),
        eps * db - 0.5 * (big_d - dm - db),
        0.515 - fi,
        0.515 - fo,
    )


# ---------------------------------------------------------- RW08 cantilever beam


def _cantilever_f(x: np.ndarray) -> float:
    return 0.0624 * float(np.sum(x))


def _cantilever_g(x: np.ndarray) -> np.ndarray:
    x1, x2, x3, x4, x5 = x
    return _arr(61.0 / x1**3 + 37.0 / x2**3 + 19.0 / x3**3 + 7.0 / x4**3 + 1.0 / x5**3 - 1.0)


# ------------------------------------------------ RW09 multiple disk clutch brake


def _clutch_repair(x: np.ndarray) -> np.ndarray:
    x = x.copy()
    x[0] = np.round(x[0])  # inner radius: 60, 61, ..., 80 mm
    x[1] = np.round(x[1])  # outer radius: 90, 91, ..., 110 mm
    x[2] = np.round(x[2] * 2.0) / 2.0  # disc thickness: 1, 1.5, ..., 3 mm
    x[3] = np.round(x[3] / 10.0) * 10.0  # actuating force: 600, 610, ..., 1000 N
    x[4] = np.round(x[4])  # number of friction surfaces: 2, ..., 9
    return x


def _clutch_f(x: np.ndarray) -> float:
    ri, ro, t, _force, z = x
    return math.pi * (ro**2 - ri**2) * t * (z + 1.0) * 0.0000078


def _clutch_g(x: np.ndarray) -> np.ndarray:
    ri, ro, t, force, z = x
    delta_r, l_max, delta, mu = 20.0, 30.0, 0.5, 0.5
    p_max, v_sr_max, t_max, safety, m_s, m_f = 1.0, 10.0, 15.0, 1.5, 40.0, 3.0
    speed, inertia = 250.0, 55.0
    ratio = (ro**3 - ri**3) / (ro**2 - ri**2)
    m_h = 2.0 / 3.0 * mu * force * z * ratio * 1.0e-3  # N m (radii in mm)
    p_rz = force / (math.pi * (ro**2 - ri**2))  # MPa
    v_sr = 2.0 * math.pi * speed * ratio / 90.0 * 1.0e-3  # m/s (radii in mm)
    stop_time = inertia * math.pi * speed / (30.0 * (m_h + m_f))
    return _arr(
        delta_r - (ro - ri),
        (z + 1.0) * (t + delta) - l_max,
        p_rz - p_max,
        p_rz * v_sr - p_max * v_sr_max,
        v_sr - v_sr_max,
        stop_time - t_max,
        safety * m_s - m_h,
        -stop_time,
    )


# ------------------------------------------------------------ RW10 step-cone pulley

_SC_SPEEDS = np.array([750.0, 450.0, 250.0, 150.0])
_SC_N, _SC_RHO, _SC_A, _SC_MU, _SC_S, _SC_T = 350.0, 7200.0, 3.0, 0.35, 1.75e6, 8.0e-3
_SC_POWER = 0.75 * 745.6998


def _pulley_f(x: np.ndarray) -> float:
    d = x[:4] * 1.0e-3
    w = x[4] * 1.0e-3
    return float(_SC_RHO * w * math.pi / 4.0 * np.sum(d**2 * (1.0 + (_SC_SPEEDS / _SC_N) ** 2)))


def _pulley_g(x: np.ndarray) -> np.ndarray:
    d = x[:4] * 1.0e-3
    w = x[4] * 1.0e-3
    ratio = _SC_SPEEDS / _SC_N
    belt = math.pi * d / 2.0 * (1.0 + ratio) + (ratio - 1.0) ** 2 * d**2 / (4.0 * _SC_A)
    belt = belt + 2.0 * _SC_A
    wrap = math.pi - 2.0 * np.arcsin(np.clip((ratio - 1.0) * d / (2.0 * _SC_A), -1.0, 1.0))
    tension_ratio = np.exp(_SC_MU * wrap)
    power = _SC_S * _SC_T * w * (1.0 - np.exp(-_SC_MU * wrap)) * math.pi * d * _SC_SPEEDS / 60.0
    equalities = np.abs(belt[0] - belt[1:]) - EQUALITY_TOL
    return np.concatenate([2.0 - tension_ratio, _SC_POWER - power, equalities])


# ------------------------------------------------------------------- registry

PROBLEMS: dict[str, EngineeringProblem] = {
    "RW01": EngineeringProblem(
        key="RW01",
        name="Tension/compression spring design",
        lower=_arr(0.05, 0.25, 2.0),
        upper=_arr(2.0, 1.3, 15.0),
        raw_objective=_spring_f,
        constraints=_spring_g,
        reference_x=_arr(0.0516890452, 0.3567173575, 11.2889881864),
        reference_f=0.012665233,
        source="Arora (1989); formulation as in Coello & Mezura-Montes (2002), Adv. Eng. "
        "Inform. 16, 193-203; best-known value, solution re-derived with SciPy SLSQP",
    ),
    "RW02": EngineeringProblem(
        key="RW02",
        name="Pressure vessel design (continuous variables)",
        lower=_arr(0.0, 0.0, 10.0, 10.0),
        upper=_arr(99.0, 99.0, 200.0, 200.0),
        raw_objective=_vessel_f,
        constraints=_vessel_g,
        reference_x=_arr(0.7781686414, 0.3846491626, 40.3196187241, 200.0),
        reference_f=5885.3327736,
        source="Sandgren (1990); continuous variant as in Kumar et al. (2020), "
        "Swarm Evol. Comput. 56, 100693",
    ),
    "RW03": EngineeringProblem(
        key="RW03",
        name="Three-bar truss design",
        lower=_arr(0.0, 0.0),
        upper=_arr(1.0, 1.0),
        raw_objective=_truss_f,
        constraints=_truss_g,
        reference_x=_arr(0.788675135, 0.408248290),
        reference_f=263.895843,
        source="Ray & Saini (2001), Eng. Optim. 33, 735-748",
    ),
    "RW04": EngineeringProblem(
        key="RW04",
        name="Welded beam design",
        lower=_arr(0.1, 0.1, 0.1, 0.1),
        upper=_arr(2.0, 10.0, 10.0, 2.0),
        raw_objective=_beam_f,
        constraints=_beam_g,
        reference_x=_arr(0.205729640, 3.470488666, 9.036623910, 0.205729640),
        reference_f=1.724852309,
        source="Ragsdell & Phillips (1976); Coello (2000), Comput. Ind. 41, 113-127",
    ),
    "RW05": EngineeringProblem(
        key="RW05",
        name="Speed reducer design",
        lower=_arr(2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0),
        upper=_arr(3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5),
        raw_objective=_reducer_f,
        constraints=_reducer_g,
        reference_x=_arr(3.5, 0.7, 17.0, 7.3, 7.715319911, 3.350214666, 5.286654465),
        reference_f=2994.471066,
        source="Golinski (1973); Mezura-Montes & Coello (2005), MICAI, LNAI 3789, 652-662",
        repair=_reducer_repair,
    ),
    "RW06": EngineeringProblem(
        key="RW06",
        name="Gear train design",
        lower=_arr(12.0, 12.0, 12.0, 12.0),
        upper=_arr(60.0, 60.0, 60.0, 60.0),
        raw_objective=_gear_f,
        constraints=_no_constraints,
        reference_x=_arr(49.0, 16.0, 19.0, 43.0),
        reference_f=2.700857e-12,
        source="Sandgren (1990), J. Mech. Des. 112, 223-229",
        repair=np.round,
    ),
    "RW07": EngineeringProblem(
        key="RW07",
        name="Rolling element bearing design",
        lower=_arr(125.0, 10.5, 4.0, 0.515, 0.515, 0.4, 0.6, 0.3, 0.02, 0.6),
        upper=_arr(150.0, 31.5, 50.0, 0.6, 0.6, 0.5, 0.7, 0.4, 0.1, 0.85),
        raw_objective=_bearing_f,
        constraints=_bearing_g,
        reference_x=_arr(
            125.7190556, 21.42559, 11.0, 0.515, 0.515, 0.4242, 0.6339, 0.3, 0.0689, 0.7995
        ),
        reference_f=-81859.74,
        source="Gupta, Tiwari & Nair (2007), Mech. Mach. Theory 42, 1418-1443; solution of "
        "Rao, Savsani & Vakharia (2011), Comput.-Aided Des. 43, 303-315",
        repair=_bearing_repair,
    ),
    "RW08": EngineeringProblem(
        key="RW08",
        name="Cantilever beam design",
        lower=_arr(0.01, 0.01, 0.01, 0.01, 0.01),
        upper=_arr(100.0, 100.0, 100.0, 100.0, 100.0),
        raw_objective=_cantilever_f,
        constraints=_cantilever_g,
        reference_x=_arr(6.01601597, 5.30917393, 4.49432960, 3.50147496, 2.15266533),
        reference_f=1.339956,
        source="Chickermane & Gea (1996), Int. J. Numer. Methods Eng. 39, 829-846",
    ),
    "RW09": EngineeringProblem(
        key="RW09",
        name="Multiple disk clutch brake design",
        lower=_arr(60.0, 90.0, 1.0, 600.0, 2.0),
        upper=_arr(80.0, 110.0, 3.0, 1000.0, 9.0),
        raw_objective=_clutch_f,
        constraints=_clutch_g,
        reference_x=_arr(70.0, 90.0, 1.0, 1000.0, 3.0),
        reference_f=0.313656611,
        source="Osyczka (2002); Deb & Srinivasan (2006), GECCO, 1629-1636",
        repair=_clutch_repair,
    ),
    "RW10": EngineeringProblem(
        key="RW10",
        name="Step-cone pulley design",
        lower=_arr(0.0, 0.0, 0.0, 0.0, 0.0),
        upper=_arr(60.0, 60.0, 90.0, 90.0, 90.0),
        raw_objective=_pulley_f,
        constraints=_pulley_g,
        reference_x=_arr(38.393739977, 52.802944687, 70.398449827, 84.495716069, 90.0),
        reference_f=16.069868725,
        source="Rao (2009); best-known value of Kumar et al. (2020), Swarm Evol. Comput. 56, "
        "100693 (solution vector re-derived with SciPy SLSQP for this repository)",
    ),
}


def get_engineering_problem(key: str) -> EngineeringProblem:
    """Engineering problem by its paper identifier (``"RW01"`` ... ``"RW10"``)."""
    normalized = key.upper()
    if normalized not in PROBLEMS:
        raise KeyError(f"unknown engineering problem {key!r}; available: {sorted(PROBLEMS)}")
    return PROBLEMS[normalized]


# Variables that are restricted to discrete values (handled by ``repair``).
DISCRETE_VARIABLES: dict[str, str] = {
    "RW05": "x3 (number of pinion teeth) is rounded to an integer",
    "RW06": "all four teeth numbers are rounded to integers",
    "RW07": "x3 (number of balls) is rounded to an integer",
    "RW09": "r_i and r_o are integers, t is a multiple of 0.5, F a multiple of 10, Z an integer",
}


def engineering_definitions() -> list[dict[str, object]]:
    """Plain-data summary of every problem (the content of ``data/engineering``)."""
    definitions = []
    for key, design in PROBLEMS.items():
        definitions.append(
            {
                "key": key,
                "name": design.name,
                "dimension": design.dim,
                "lower_bounds": [float(v) for v in design.lower],
                "upper_bounds": [float(v) for v in design.upper],
                "n_constraints": int(design.constraints(design.reference_x).size),
                "discrete_variables": DISCRETE_VARIABLES.get(key, "none"),
                "reference_solution": [float(v) for v in design.reference_x],
                "reference_objective": float(design.reference_f),
                "source": design.source,
            }
        )
    return definitions
