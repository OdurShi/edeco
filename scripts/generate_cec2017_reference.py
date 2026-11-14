#!/usr/bin/env python
"""Regenerate ``tests/data/cec2017_reference.json`` from the official CEC 2017 C code.

This is a maintenance tool; it is not needed to use the package.  It compiles the
organisers' ``cec17_func.cpp`` (not distributed with this repository) into a small
command-line program, evaluates the deterministic probe points of
``edeco.benchmarks.cec2017.validation_points`` and stores the returned values, which
``tests/test_cec2017.py`` compares against the NumPy implementation.

Example::

    python scripts/generate_cec2017_reference.py \
        --source /path/to/cec17_func.cpp --data-root /path/to/folder_containing_input_data
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from edeco.benchmarks import cec2017

MEX_STUB = """
#ifndef MEX_STUB_H
#define MEX_STUB_H
#include <stdlib.h>
#include <stdio.h>
typedef struct mxArray_tag { int unused; } mxArray;
#define mxREAL 0
static inline int mexPrintf(const char *fmt, ...) { (void)fmt; return 0; }
static inline void mexErrMsgTxt(const char *msg) { fprintf(stderr, "%s\\n", msg); exit(1); }
static inline size_t mxGetM(const mxArray *a) { (void)a; return 0; }
static inline size_t mxGetN(const mxArray *a) { (void)a; return 0; }
static inline double *mxGetPr(const mxArray *a) { (void)a; return NULL; }
static inline mxArray *mxCreateDoubleMatrix(size_t m, size_t n, int c) {
  (void)m; (void)n; (void)c; return NULL;
}
#endif
"""

HARNESS = """
#include "{source}"
#include <vector>
int main() {{
  int func, dim, n;
  while (scanf("%d %d %d", &func, &dim, &n) == 3) {{
    std::vector<double> x(dim * n), f(n);
    for (int i = 0; i < dim * n; i++) if (scanf("%lf", &x[i]) != 1) return 2;
    cec17_test_func(x.data(), f.data(), dim, n, func);
    for (int i = 0; i < n; i++) printf("%d %d %.17g\\n", func, dim, f[i]);
  }}
  return 0;
}}
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="official cec17_func.cpp")
    parser.add_argument(
        "--data-root", type=Path, required=True, help="directory that contains input_data/"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "tests" / "data" / "cec2017_reference.json",
    )
    parser.add_argument("--compiler", default="c++")
    args = parser.parse_args()

    data_dir = args.data_root / "input_data"
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        (work / "mex.h").write_text(MEX_STUB)
        (work / "malloc.h").write_text("#include <stdlib.h>\n")
        (work / "harness.cpp").write_text(HARNESS.format(source=args.source.resolve()))
        binary = work / "cec17"
        subprocess.run(
            [args.compiler, "-O2", "-w", "-std=c++14", f"-I{work}", "harness.cpp", "-o", binary],
            cwd=work,
            check=True,
        )
        requests = []
        keys = []
        for dim in cec2017.SUPPORTED_DIMS:
            for func_num in cec2017.FUNCTION_IDS:
                function = cec2017.CEC2017Function(func_num, dim, data_dir)
                points = cec2017.validation_points(func_num, dim, function.x_opt)
                rows = "\n".join(" ".join(f"{v:.17g}" for v in row) for row in points)
                requests.append(f"{func_num} {dim} {len(points)}\n{rows}")
                keys.append((func_num, dim, len(points)))
        completed = subprocess.run(
            [binary],
            input="\n".join(requests) + "\n",
            capture_output=True,
            text=True,
            cwd=args.data_root,
            check=True,
        )
    values = [line.split() for line in completed.stdout.splitlines() if len(line.split()) == 3]
    reference: dict[str, dict[str, list[float]]] = {}
    cursor = 0
    for func_num, dim, n_points in keys:
        chunk = values[cursor : cursor + n_points]
        cursor += n_points
        assert all(int(v[0]) == func_num and int(v[1]) == dim for v in chunk)
        reference.setdefault(str(dim), {})[str(func_num)] = [float(v[2]) for v in chunk]
    payload = {
        "description": (
            "Values returned by the official CEC 2017 C implementation (cec17_func.cpp, "
            "Awad et al.) at edeco.benchmarks.cec2017.validation_points(func, dim, x_opt)."
        ),
        "values": reference,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote {cursor} reference values to {args.output}")


if __name__ == "__main__":
    main()
