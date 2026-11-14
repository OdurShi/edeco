"""Experiment configuration (one YAML file per paper experiment in ``configs/``)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from edeco.benchmarks import SUITES, list_functions
from edeco.optimizer import ECOConfig, get_config


@dataclass(frozen=True)
class ExperimentConfig:
    """Protocol of one experiment.

    Attributes
    ----------
    name:
        Experiment identifier (used for the default output directory).
    suite:
        ``"cec2017"``, ``"classical"`` or ``"engineering"``.
    functions:
        Function identifiers; ``"all"`` selects the whole suite.
    dim:
        Problem dimension (ignored by the engineering suite, whose problems have fixed
        dimensions).
    algorithms:
        Either a list of registered names (``ECO``, ``EECO``, ``DECO``, ``EDECO``) or a
        mapping ``label -> options`` where ``options`` may contain ``base`` (a registered
        name, default: the label), ``pop_size`` (population size of this entry only) and
        any :class:`edeco.ECOConfig` field to override.
    pop_size:
        Population size ``N``.  The paper does not state it; see
        ``docs/implementation_notes.md`` for the choice of 100.
    fes_per_dim:
        Evaluation budget per dimension; the paper uses ``Fes_max = 5000 * D``.
    max_fes:
        Absolute evaluation budget; overrides ``fes_per_dim`` when given.
    runs:
        Number of independent runs per (function, algorithm); 30 in the paper.
    seed:
        Root seed.  Run ``r`` on function ``f`` uses the same seed for every algorithm,
        so all algorithms start from the same initial population.
    workers:
        Number of worker processes.
    history_points:
        Number of checkpoints at which the best-so-far value is stored.
    penalty:
        Penalty coefficient of the engineering suite (``None``: package default).
    """

    name: str
    suite: str
    functions: list[str | int] | str = "all"
    dim: int | None = None
    algorithms: list[str] | dict[str, dict[str, Any]] = field(
        default_factory=lambda: ["ECO", "EECO", "DECO", "EDECO"]
    )
    pop_size: int = 100
    fes_per_dim: int = 5000
    max_fes: int | None = None
    runs: int = 30
    seed: int = 2025
    workers: int = 1
    history_points: int = 50
    penalty: float | None = None

    def __post_init__(self) -> None:
        if self.suite not in SUITES:
            raise ValueError(f"suite must be one of {SUITES}, got {self.suite!r}")
        if self.suite != "engineering" and self.dim is None:
            raise ValueError(f"suite {self.suite!r} requires 'dim'")
        if self.runs < 1 or self.workers < 1 or self.history_points < 2:
            raise ValueError("runs and workers must be >= 1 and history_points >= 2")

    def function_list(self) -> list[str | int]:
        """Resolved list of function identifiers."""
        if isinstance(self.functions, str):
            if self.functions != "all":
                raise ValueError("functions must be a list or the string 'all'")
            return list_functions(self.suite)
        return list(self.functions)

    def algorithm_configs(self) -> dict[str, ECOConfig]:
        """Resolved ``label -> ECOConfig`` mapping, in the order given."""
        if isinstance(self.algorithms, dict):
            resolved = {}
            for label, options in self.algorithms.items():
                options = dict(options or {})
                base = options.pop("base", label)
                options.pop("pop_size", None)
                resolved[str(label)] = get_config(base, **options)
            return resolved
        return {str(name): get_config(name) for name in self.algorithms}

    def population_sizes(self) -> dict[str, int]:
        """Population size of every algorithm entry (``pop_size`` unless overridden)."""
        sizes = dict.fromkeys(self.algorithm_configs(), self.pop_size)
        if isinstance(self.algorithms, dict):
            for label, options in self.algorithms.items():
                if options and "pop_size" in options:
                    sizes[str(label)] = int(options["pop_size"])
        return sizes

    def budget(self, dim: int) -> int:
        """Evaluation budget for a problem of dimension ``dim``."""
        return int(self.max_fes) if self.max_fes is not None else int(self.fes_per_dim * dim)

    def to_dict(self) -> dict[str, Any]:
        """Plain-dictionary form (for ``config.json``)."""
        return asdict(self)


def load_config(path: str | Path, **overrides: Any) -> ExperimentConfig:
    """Read an experiment YAML file; keyword ``overrides`` replace individual fields."""
    with open(path, encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    data.update({key: value for key, value in overrides.items() if value is not None})
    data.setdefault("name", Path(path).stem)
    return ExperimentConfig(**data)
