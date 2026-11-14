"""CEC 2017 single-objective bound-constrained test suite (F1, F3-F30).

The paper evaluates on "29 CEC 2017 benchmark functions" in 10, 30, 50 and 100
dimensions (Table 1); F2 is excluded, as in the official suite, because it was deleted
by the organisers.

This module is a NumPy implementation of the official problem definitions (Awad, Ali,
Liang, Qu and Suganthan, 2016) and reproduces the behaviour of the organisers' C
implementation, including two of its well-known quirks: F6 ignores its rotation matrix
and F8 does not apply the rounding step of the non-continuous Rastrigin function.  It
was validated against a locally compiled copy of the official C code; the reference
values are stored in ``tests/data/cec2017_reference.json``.

The shift vectors, rotation matrices and shuffle permutations of the organisers are
shipped with the repository as one NumPy archive per dimension (``data/cec2017``, built
by ``scripts/build_cec2017_data.py``).  A different location can be given with the
``data_dir`` argument or the ``EDECO_CEC2017_DATA`` environment variable; it may hold
either these archives or a copy of the official ``input_data`` text files.  When no
data directory is found, the text files bundled with the optional ``opfunu`` package are
used.

Note that the CEC 2017 classes of ``opfunu`` 1.0.4 themselves are *not* used: they
renumber the functions (its F2-F29 are the official F3-F30 with biases shifted by 100),
load the data files of the preceding function for the official F3-F20 and differ
numerically from the official code for most functions.  See
``docs/implementation_notes.md``.
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Callable, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

SUPPORTED_DIMS = (10, 30, 50, 100)
FUNCTION_IDS = (1, *range(3, 31))
LOWER_BOUND, UPPER_BOUND = -100.0, 100.0
DATA_ENV_VAR = "EDECO_CEC2017_DATA"

_INF = 1.0e99
_TWO_PI = 2.0 * math.pi


class CEC2017DataError(RuntimeError):
    """Raised when the official CEC 2017 data files cannot be located."""


def _has_data(directory: Path, dim: int | None = None) -> bool:
    """``True`` when ``directory`` holds the archives or the official text files."""
    archive = f"cec2017_D{dim}.npz" if dim is not None else f"cec2017_D{SUPPORTED_DIMS[0]}.npz"
    return (directory / archive).is_file() or (directory / "shift_data_1.txt").is_file()


def find_data_dir(data_dir: str | os.PathLike[str] | None = None) -> Path:
    """Locate the directory holding the CEC 2017 definition data.

    Search order: the explicit argument; the ``EDECO_CEC2017_DATA`` environment variable;
    ``data/cec2017`` of the repository this module belongs to; ``data/cec2017`` below the
    current working directory; the ``data_2017`` directory of an installed ``opfunu``
    package.  An explicit argument or environment variable is not overridden by the later
    entries.
    """
    if data_dir is not None:
        candidates = [Path(data_dir)]
    elif os.environ.get(DATA_ENV_VAR):
        candidates = [Path(os.environ[DATA_ENV_VAR])]
    else:
        candidates = [
            Path(__file__).resolve().parents[3] / "data" / "cec2017",
            Path.cwd() / "data" / "cec2017",
        ]
        from importlib.util import find_spec

        spec = find_spec("opfunu")
        if spec is not None and spec.origin is not None:
            candidates.append(Path(spec.origin).parent / "cec_based" / "data_2017")
    for candidate in candidates:
        if _has_data(candidate):
            return candidate
    raise CEC2017DataError(
        "CEC 2017 data not found. Run from a clone of the repository (data/cec2017), set "
        f"the {DATA_ENV_VAR} environment variable to a directory with the archives or the "
        "official 'input_data' files, or install the optional dependency "
        "(pip install 'edeco[cec]')."
    )


def is_available(data_dir: str | os.PathLike[str] | None = None) -> bool:
    """``True`` when the CEC 2017 definition data can be found."""
    try:
        find_data_dir(data_dir)
    except CEC2017DataError:
        return False
    return True


def read_official_files(
    directory: Path, func_num: int, dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Parse the organisers' text files of one function.

    Returns ``(rotation, shift, shuffle)`` with shapes ``(parts, D, D)``, ``(parts, D)``
    and ``(parts, D)`` (1-based indices; ``None`` for functions without shuffling), where
    ``parts`` is 10 for the composition functions F21-F30 and 1 otherwise.
    """
    n_parts = 10 if func_num > 20 else 1
    rotation = np.array((directory / f"M_{func_num}_D{dim}.txt").read_text().split(), dtype=float)
    rotation = rotation[: n_parts * dim * dim].reshape(n_parts, dim, dim)
    rows = (directory / f"shift_data_{func_num}.txt").read_text().splitlines()
    rows = [row for row in rows if row.strip()][:n_parts]
    shift = np.array([np.array(row.split()[:dim], dtype=float) for row in rows])
    shuffle = None
    if 11 <= func_num <= 20 or func_num >= 29:
        text = (directory / f"shuffle_data_{func_num}_D{dim}.txt").read_text()
        shuffle = np.array(text.split(), dtype=float)[: n_parts * dim]
        shuffle = shuffle.astype(int).reshape(n_parts, dim)
    return rotation, shift, shuffle


def load_definition(
    directory: Path, func_num: int, dim: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Rotation matrices, shift vectors and shuffle indices of one function.

    Reads ``cec2017_D<D>.npz`` when the directory contains it and the official text files
    otherwise; both give identical arrays.
    """
    archive_path = directory / f"cec2017_D{dim}.npz"
    if not archive_path.is_file():
        return read_official_files(directory, func_num, dim)
    with np.load(archive_path) as archive:
        rotation = archive[f"M_{func_num}"]
        shift = archive[f"shift_{func_num}"]
        key = f"shuffle_{func_num}"
        shuffle = archive[key].astype(int) if key in archive.files else None
    return rotation, shift, shuffle


# ------------------------------------------------------------------ basic functions
# Each takes the already shifted / scaled / rotated vector ``z`` of its own segment.


def _ellips(z: np.ndarray) -> float:
    n = z.size
    return float(np.sum(10.0 ** (6.0 * np.arange(n) / (n - 1)) * z * z))


def _bent_cigar(z: np.ndarray) -> float:
    return float(z[0] * z[0] + 1.0e6 * np.sum(z[1:] * z[1:]))


def _discus(z: np.ndarray) -> float:
    return float(1.0e6 * z[0] * z[0] + np.sum(z[1:] * z[1:]))


def _zakharov(z: np.ndarray) -> float:
    sum1 = float(np.sum(z * z))
    sum2 = float(np.sum(0.5 * np.arange(1, z.size + 1) * z))
    return sum1 + sum2**2 + sum2**4


def _rosenbrock(z: np.ndarray) -> float:
    z = z + 1.0
    tmp1 = z[:-1] * z[:-1] - z[1:]
    tmp2 = z[:-1] - 1.0
    return float(np.sum(100.0 * tmp1 * tmp1 + tmp2 * tmp2))


def _ackley(z: np.ndarray) -> float:
    n = z.size
    sum1 = -0.2 * math.sqrt(float(np.sum(z * z)) / n)
    sum2 = float(np.sum(np.cos(_TWO_PI * z))) / n
    return math.e - 20.0 * math.exp(sum1) - math.exp(sum2) + 20.0


_WEIERSTRASS_A = 0.5 ** np.arange(21)
_WEIERSTRASS_B = 3.0 ** np.arange(21)


def _weierstrass(z: np.ndarray) -> float:
    terms = _WEIERSTRASS_A * np.cos(_TWO_PI * _WEIERSTRASS_B * (z[:, None] + 0.5))
    sum2 = float(np.sum(_WEIERSTRASS_A * np.cos(_TWO_PI * _WEIERSTRASS_B * 0.5)))
    return float(np.sum(terms)) - z.size * sum2


def _griewank(z: np.ndarray) -> float:
    s = float(np.sum(z * z))
    p = float(np.prod(np.cos(z / np.sqrt(1.0 + np.arange(z.size)))))
    return 1.0 + s / 4000.0 - p


def _rastrigin(z: np.ndarray) -> float:
    return float(np.sum(z * z - 10.0 * np.cos(_TWO_PI * z) + 10.0))


def _schwefel(z: np.ndarray) -> float:
    n = z.size
    z = z + 4.209687462275036e002
    total = 0.0
    high = z > 500.0
    low = z < -500.0
    mid = ~(high | low)
    if high.any():
        zh = z[high]
        m = 500.0 - np.fmod(zh, 500.0)
        total -= float(np.sum(m * np.sin(np.sqrt(m))))
        total += float(np.sum(((zh - 500.0) / 100.0) ** 2)) / n
    if low.any():
        zl = z[low]
        m = np.fmod(np.abs(zl), 500.0)
        total -= float(np.sum((-500.0 + m) * np.sin(np.sqrt(500.0 - m))))
        total += float(np.sum(((zl + 500.0) / 100.0) ** 2)) / n
    if mid.any():
        zm = z[mid]
        total -= float(np.sum(zm * np.sin(np.sqrt(np.abs(zm)))))
    return total + 4.189828872724338e002 * n


_KATSUURA_POW = 2.0 ** np.arange(1, 33)


def _katsuura(z: np.ndarray) -> float:
    n = z.size
    tmp3 = float(n) ** 1.2
    scaled = _KATSUURA_POW * z[:, None]
    temp = np.sum(np.abs(scaled - np.floor(scaled + 0.5)) / _KATSUURA_POW, axis=1)
    f = float(np.prod((1.0 + np.arange(1, n + 1) * temp) ** (10.0 / tmp3)))
    tmp1 = 10.0 / n / n
    return f * tmp1 - tmp1


def _grie_rosen(z: np.ndarray) -> float:
    z = z + 1.0
    nxt = np.roll(z, -1)
    tmp1 = z * z - nxt
    tmp2 = z - 1.0
    temp = 100.0 * tmp1 * tmp1 + tmp2 * tmp2
    return float(np.sum(temp * temp / 4000.0 - np.cos(temp) + 1.0))


def _escaffer6(z: np.ndarray) -> float:
    nxt = np.roll(z, -1)
    sq = z * z + nxt * nxt
    temp1 = np.sin(np.sqrt(sq)) ** 2
    temp2 = 1.0 + 0.001 * sq
    return float(np.sum(0.5 + (temp1 - 0.5) / (temp2 * temp2)))


def _happycat(z: np.ndarray) -> float:
    n = z.size
    z = z - 1.0
    r2 = float(np.sum(z * z))
    sum_z = float(np.sum(z))
    return abs(r2 - n) ** (2.0 * 0.125) + (0.5 * r2 + sum_z) / n + 0.5


def _hgbat(z: np.ndarray) -> float:
    n = z.size
    z = z - 1.0
    r2 = float(np.sum(z * z))
    sum_z = float(np.sum(z))
    return abs(r2**2 - sum_z**2) ** (2.0 * 0.25) + (0.5 * r2 + sum_z) / n + 0.5


def _levy(z: np.ndarray) -> float:
    w = 1.0 + (z - 1.0) / 4.0
    term1 = math.sin(math.pi * w[0]) ** 2
    term3 = (w[-1] - 1.0) ** 2 * (1.0 + math.sin(_TWO_PI * w[-1]) ** 2)
    wi = w[:-1]
    total = float(np.sum((wi - 1.0) ** 2 * (1.0 + 10.0 * np.sin(math.pi * wi + 1.0) ** 2)))
    return term1 + total + term3


def _schaffer_f7(y: np.ndarray) -> float:
    n = y.size
    s = np.sqrt(y[:-1] * y[:-1] + y[1:] * y[1:])
    tmp = np.sin(50.0 * s**0.2)
    f = float(np.sum(np.sqrt(s) + np.sqrt(s) * tmp * tmp))
    return f * f / (n - 1) / (n - 1)


def _bi_rastrigin(
    y: np.ndarray, shift_sign_source: np.ndarray, rotation: np.ndarray | None
) -> float:
    """Lunacek bi-Rastrigin on the shifted (not rotated) vector ``y``."""
    n = y.size
    mu0, d = 2.5, 1.0
    s = 1.0 - 1.0 / (2.0 * math.sqrt(n + 20.0) - 8.2)
    mu1 = -math.sqrt((mu0 * mu0 - d) / s)
    tmpx = 2.0 * (y * (10.0 / 100.0))
    tmpx = np.where(shift_sign_source < 0.0, -tmpx, tmpx)
    z = tmpx
    tmpx = tmpx + mu0
    tmp1 = float(np.sum((tmpx - mu0) ** 2))
    tmp2 = s * float(np.sum((tmpx - mu1) ** 2)) + d * n
    angle = rotation @ z if rotation is not None else z
    tmp = float(np.sum(np.cos(_TWO_PI * angle)))
    return min(tmp1, tmp2) + 10.0 * (n - tmp)


# name -> (function of z, scaling applied before rotation)
_BASIC: dict[str, tuple[Callable[[np.ndarray], float], float]] = {
    "ellips": (_ellips, 1.0),
    "bent_cigar": (_bent_cigar, 1.0),
    "discus": (_discus, 1.0),
    "zakharov": (_zakharov, 1.0),
    "rosenbrock": (_rosenbrock, 2.048 / 100.0),
    "ackley": (_ackley, 1.0),
    "weierstrass": (_weierstrass, 0.5 / 100.0),
    "griewank": (_griewank, 600.0 / 100.0),
    "rastrigin": (_rastrigin, 5.12 / 100.0),
    "schwefel": (_schwefel, 1000.0 / 100.0),
    "katsuura": (_katsuura, 5.0 / 100.0),
    "grie_rosen": (_grie_rosen, 5.0 / 100.0),
    "escaffer6": (_escaffer6, 1.0),
    "happycat": (_happycat, 5.0 / 100.0),
    "hgbat": (_hgbat, 5.0 / 100.0),
    "levy": (_levy, 1.0),
}

_SIMPLE = {
    1: "bent_cigar",
    3: "zakharov",
    4: "rosenbrock",
    5: "rastrigin",
    8: "rastrigin",  # official code: the non-continuous rounding step has no effect
    9: "levy",
    10: "schwefel",
}

# hybrid function index -> (group proportions, component names)
_HYBRID: dict[int, tuple[tuple[float, ...], tuple[str, ...]]] = {
    1: ((0.2, 0.4, 0.4), ("zakharov", "rosenbrock", "rastrigin")),
    2: ((0.3, 0.3, 0.4), ("ellips", "schwefel", "bent_cigar")),
    3: ((0.3, 0.3, 0.4), ("bent_cigar", "rosenbrock", "bi_rastrigin")),
    4: ((0.2, 0.2, 0.2, 0.4), ("ellips", "ackley", "schaffer_f7", "rastrigin")),
    5: ((0.2, 0.2, 0.3, 0.3), ("bent_cigar", "hgbat", "rastrigin", "rosenbrock")),
    6: ((0.2, 0.2, 0.3, 0.3), ("escaffer6", "hgbat", "rosenbrock", "schwefel")),
    7: ((0.1, 0.2, 0.2, 0.2, 0.3), ("katsuura", "ackley", "grie_rosen", "schwefel", "rastrigin")),
    8: ((0.2, 0.2, 0.2, 0.2, 0.2), ("ellips", "ackley", "rastrigin", "hgbat", "discus")),
    9: (
        (0.2, 0.2, 0.2, 0.2, 0.2),
        ("bent_cigar", "rastrigin", "grie_rosen", "weierstrass", "escaffer6"),
    ),
    10: (
        (0.1, 0.1, 0.2, 0.2, 0.2, 0.2),
        ("hgbat", "katsuura", "ackley", "rastrigin", "schwefel", "schaffer_f7"),
    ),
}

# composition function index -> (sigmas, [(component, multiplier, divisor), ...])
_COMPOSITION: dict[int, tuple[tuple[float, ...], tuple[tuple[str, float, float], ...]]] = {
    1: ((10, 20, 30), (("rosenbrock", 1, 1), ("ellips", 10000, 1e10), ("rastrigin", 1, 1))),
    2: ((10, 20, 30), (("rastrigin", 1, 1), ("griewank", 1000, 100), ("schwefel", 1, 1))),
    3: (
        (10, 20, 30, 40),
        (("rosenbrock", 1, 1), ("ackley", 1000, 100), ("schwefel", 1, 1), ("rastrigin", 1, 1)),
    ),
    4: (
        (10, 20, 30, 40),
        (
            ("ackley", 1000, 100),
            ("ellips", 10000, 1e10),
            ("griewank", 1000, 100),
            ("rastrigin", 1, 1),
        ),
    ),
    5: (
        (10, 20, 30, 40, 50),
        (
            ("rastrigin", 10000, 1e3),
            ("happycat", 1000, 1e3),
            ("ackley", 1000, 100),
            ("discus", 10000, 1e10),
            ("rosenbrock", 1, 1),
        ),
    ),
    6: (
        (10, 20, 20, 30, 40),
        (
            ("escaffer6", 10000, 2e7),
            ("schwefel", 1, 1),
            ("griewank", 1000, 100),
            ("rosenbrock", 1, 1),
            ("rastrigin", 10000, 1e3),
        ),
    ),
    7: (
        (10, 20, 30, 40, 50, 60),
        (
            ("hgbat", 10000, 1000),
            ("rastrigin", 10000, 1e3),
            ("schwefel", 10000, 4e3),
            ("bent_cigar", 10000, 1e30),
            ("ellips", 10000, 1e10),
            ("escaffer6", 10000, 2e7),
        ),
    ),
    8: (
        (10, 20, 30, 40, 50, 60),
        (
            ("ackley", 1000, 100),
            ("griewank", 1000, 100),
            ("discus", 10000, 1e10),
            ("rosenbrock", 1, 1),
            ("happycat", 1000, 1e3),
            ("escaffer6", 10000, 2e7),
        ),
    ),
}

# compositions 9 and 10 combine three hybrid functions
_COMPOSITION_OF_HYBRIDS = {9: ((10, 30, 50), (5, 6, 7)), 10: ((10, 30, 50), (5, 8, 9))}


def _group_slices(proportions: Sequence[float], n: int) -> list[slice]:
    """Segment boundaries of a hybrid function (``ceil`` of the proportions, as in C)."""
    sizes = [math.ceil(p * n) for p in proportions[:-1]]
    sizes.append(n - sum(sizes))
    bounds = np.concatenate([[0], np.cumsum(sizes)])
    return [slice(int(a), int(b)) for a, b in itertools.pairwise(bounds)]


class CEC2017Function:
    """One function of the CEC 2017 suite.

    Parameters
    ----------
    func_num:
        Official function number (1, 3, 4, ..., 30).
    dim:
        Problem dimension: 10, 30, 50 or 100.
    data_dir:
        Optional directory with the definition data (see :func:`find_data_dir`).
    """

    def __init__(
        self, func_num: int, dim: int, data_dir: str | os.PathLike[str] | None = None
    ) -> None:
        if func_num not in FUNCTION_IDS:
            raise ValueError(f"CEC 2017 function number must be one of {FUNCTION_IDS}")
        if dim not in SUPPORTED_DIMS:
            raise ValueError(f"CEC 2017 functions are defined for D in {SUPPORTED_DIMS}")
        self.func_num = func_num
        self.dim = dim
        self.f_bias = 100.0 * func_num
        rotation, shift, shuffle = load_definition(find_data_dir(data_dir), func_num, dim)
        self._rotation = rotation
        self._shift = shift
        self._shuffle = None if shuffle is None else shuffle - 1  # zero-based indices

    @property
    def x_opt(self) -> np.ndarray:
        """Location of the global optimum.

        It is the (first) shift vector, except for F9: the Levy function of the official
        code attains its minimum where the rotated vector equals one in every coordinate,
        i.e. at ``o + M^-1 1``.
        """
        if self.func_num == 9:
            return self._shift[0] + np.linalg.solve(self._rotation[0], np.ones(self.dim))
        return self._shift[0].copy()

    @property
    def f_opt(self) -> float:
        """Global optimum value ``F_i* = 100 * i``."""
        return self.f_bias

    # ------------------------------------------------------------------ evaluation

    def _basic(self, name: str, x: np.ndarray, part: int) -> float:
        """A basic function applied to ``x`` with shift and rotation number ``part``."""
        func, rate = _BASIC[name]
        return func(self._rotation[part] @ ((x - self._shift[part]) * rate))

    def _hybrid(self, index: int, x: np.ndarray, part: int) -> float:
        """Hybrid function ``index`` using shift / rotation / shuffle number ``part``."""
        proportions, components = _HYBRID[index]
        assert self._shuffle is not None
        shift = self._shift[part]
        z = self._rotation[part] @ (x - shift)
        y = z[self._shuffle[part]]
        total = 0.0
        for name, segment in zip(components, _group_slices(proportions, self.dim), strict=True):
            n_segment = segment.stop - segment.start
            if name == "schaffer_f7":
                # Official code: this component reads the first n_segment entries of the
                # permuted vector instead of its own segment.
                total += _schaffer_f7(y[:n_segment])
            elif name == "bi_rastrigin":
                # Official code: the sign pattern comes from the first entries of the shift.
                total += _bi_rastrigin(y[segment], shift[:n_segment], None)
            else:
                func, rate = _BASIC[name]
                total += func(y[segment] * rate)
        return total

    def _compose(self, x: np.ndarray, sigmas: Sequence[float], fits: Sequence[float]) -> float:
        """Weighted combination ``cf_cal`` of the composition functions."""
        n_parts = len(sigmas)
        weights = np.empty(n_parts)
        values = np.empty(n_parts)
        for i in range(n_parts):
            values[i] = fits[i] + 100.0 * i
            dist2 = float(np.sum((x - self._shift[i]) ** 2))
            if dist2 != 0.0:
                weights[i] = (1.0 / dist2) ** 0.5 * math.exp(
                    -dist2 / 2.0 / self.dim / sigmas[i] ** 2
                )
            else:
                weights[i] = _INF
        total = float(np.sum(weights))
        if weights.max() == 0.0:
            weights[:] = 1.0
            total = float(n_parts)
        return float(np.sum(weights / total * values))

    def __call__(self, x: np.ndarray) -> float:
        """Objective value ``F_i(x)`` including the bias ``100 * i``."""
        x = np.asarray(x, dtype=float)
        if x.shape != (self.dim,):
            raise ValueError(f"expected a vector of length {self.dim}, got shape {x.shape}")
        k = self.func_num
        if k in _SIMPLE:
            value = self._basic(_SIMPLE[k], x, 0)
        elif k == 6:
            value = _schaffer_f7(x - self._shift[0])  # official code: rotation not applied
        elif k == 7:
            value = _bi_rastrigin(x - self._shift[0], self._shift[0], self._rotation[0])
        elif k <= 20:
            value = self._hybrid(k - 10, x, 0)
        elif k <= 28:
            sigmas, parts = _COMPOSITION[k - 20]
            fits = [
                mult * self._basic(name, x, i) / div for i, (name, mult, div) in enumerate(parts)
            ]
            value = self._compose(x, sigmas, fits)
        else:
            sigmas, hybrids = _COMPOSITION_OF_HYBRIDS[k - 20]
            fits = [self._hybrid(index, x, i) for i, index in enumerate(hybrids)]
            value = self._compose(x, sigmas, fits)
        return value + self.f_bias


@lru_cache(maxsize=64)
def _cached(func_num: int, dim: int, data_dir: str | None) -> CEC2017Function:
    return CEC2017Function(func_num, dim, data_dir)


def make_function(
    func_num: int, dim: int, data_dir: str | os.PathLike[str] | None = None
) -> CEC2017Function:
    """Cached constructor of :class:`CEC2017Function`."""
    return _cached(func_num, dim, None if data_dir is None else str(data_dir))


def validation_points(func_num: int, dim: int, x_opt: np.ndarray) -> np.ndarray:
    """Deterministic probe points used to validate against the official C implementation.

    Four points per function: two spread over the search range and two close to the
    global optimum.  They are generated by closed-form expressions (no random numbers) so
    the stored reference values in ``tests/data/cec2017_reference.json`` stay meaningful
    on every platform.
    """
    idx = np.arange(dim, dtype=float)
    far = 90.0 * np.sin(0.7 * (idx + 1.0) * (func_num + 1.0) + 0.3)
    mid = 60.0 * np.cos(1.3 * (idx + 1.0) + func_num)
    near = x_opt + 2.0 * np.sin(idx + func_num)
    very_near = x_opt + 0.05 * np.cos(2.0 * idx + func_num)
    return np.clip(np.vstack([far, mid, near, very_near]), LOWER_BOUND, UPPER_BOUND)
