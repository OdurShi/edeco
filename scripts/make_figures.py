#!/usr/bin/env python
"""Write the figures of an experiment (PNG) to ``<results>/figures``.

Thin wrapper around ``edeco figures``::

    python scripts/make_figures.py --results results/cec2017_10d

Produces convergence curves, box plots, the Friedman ranking with the Nemenyi critical
difference (information of Figs. 3, 6, 7 and 8) and the Wilcoxon rank-sum counts (Fig. 5).
"""

from __future__ import annotations

import sys

from edeco.cli import main

if __name__ == "__main__":
    sys.exit(main(["figures", *sys.argv[1:]]))
