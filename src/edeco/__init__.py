"""EDECO: an Enhanced Educational Competition Optimizer (Python re-implementation).

Public entry points::

    from edeco import minimize, get_config

    result = minimize(objective, (-100.0, 100.0), dim=10, pop_size=30, max_fes=50_000,
                      seed=0, config=get_config("EDECO"))
"""

from edeco.evaluator import BudgetExhaustedError, Evaluator
from edeco.optimizer import ALGORITHMS, ECOConfig, OptimizeResult, get_config, minimize

__version__ = "1.0.0"

__all__ = [
    "ALGORITHMS",
    "BudgetExhaustedError",
    "ECOConfig",
    "Evaluator",
    "OptimizeResult",
    "__version__",
    "get_config",
    "minimize",
]
