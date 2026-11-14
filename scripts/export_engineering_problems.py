#!/usr/bin/env python
"""Write the definitions of the ten engineering problems to ``data/engineering``.

``data/engineering/problems.json`` is a machine-readable summary of the problems coded
in ``src/edeco/benchmarks/engineering.py`` (variables, bounds, number of constraints,
reference solution and value, literature source).  ``tests/test_data.py`` checks that the
file agrees with the code.

    python scripts/export_engineering_problems.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edeco.benchmarks.engineering import (
    DEFAULT_PENALTY,
    EQUALITY_TOL,
    FEASIBILITY_TOL,
    engineering_definitions,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "engineering" / "problems.json",
    )
    args = parser.parse_args()
    payload = {
        "description": "Constrained engineering design problems RW01-RW10 (Table 7 of the paper)",
        "constraint_handling": {
            "method": "static quadratic penalty f(x) + rho * sum(max(0, g_i(x))^2)",
            "rho": DEFAULT_PENALTY,
            "equality_tolerance": EQUALITY_TOL,
            "feasibility_tolerance": FEASIBILITY_TOL,
        },
        "problems": engineering_definitions(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(payload['problems'])} problem definitions to {args.output}")


if __name__ == "__main__":
    main()
