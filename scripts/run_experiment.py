#!/usr/bin/env python
"""Run an experiment described by a YAML file in ``configs/``.

Thin wrapper around ``edeco run``::

    python scripts/run_experiment.py --config configs/cec2017_10d.yaml --output results/cec2017_10d

Optional overrides: ``--runs``, ``--workers``, ``--seed``, ``--functions``.
"""

from __future__ import annotations

import sys

from edeco.cli import main

if __name__ == "__main__":
    sys.exit(main(["--verbose", "run", *sys.argv[1:]]))
