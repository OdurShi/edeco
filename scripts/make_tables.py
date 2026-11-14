#!/usr/bin/env python
"""Write the result tables of an experiment (Markdown and CSV) to ``<results>/tables``.

Thin wrapper around ``edeco tables``::

    python scripts/make_tables.py --results results/cec2017_10d

Produces ``summary`` (layout of Tables A1-A4), ``friedman`` (Tables 3 and 6),
``rank_sum`` and ``rank_sum_detail`` (Tables 4 and 5) and, for the engineering suite,
``engineering`` (Table 8).
"""

from __future__ import annotations

import sys

from edeco.cli import main

if __name__ == "__main__":
    sys.exit(main(["tables", *sys.argv[1:]]))
