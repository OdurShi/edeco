#!/usr/bin/env python
"""Convert the official CEC 2017 ``input_data`` text files into ``data/cec2017/*.npz``.

This is a maintenance tool: the archives it writes are already part of the repository.
It reads the organisers' files (``M_<f>_D<D>.txt``, ``shift_data_<f>.txt``,
``shuffle_data_<f>_D<D>.txt``) for the dimensions used in the paper and stores, per
dimension, one compressed NumPy archive with the arrays

* ``M_<f>``        rotation matrices, shape ``(parts, D, D)``,
* ``shift_<f>``    shift vectors, shape ``(parts, D)``,
* ``shuffle_<f>``  shuffle permutations (1-based, as in the official files),
  shape ``(parts, D)``, for the hybrid functions and the compositions built on them,

where ``parts`` is 10 for the composition functions F21-F30 and 1 otherwise.  The numeric
values are those obtained by parsing the text files as 64-bit floats, i.e. exactly what a
loader of the text files would use.

Example::

    python scripts/build_cec2017_data.py --input-dir /path/to/input_data
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

from edeco.benchmarks import cec2017


def build(input_dir: Path, output_dir: Path) -> list[Path]:
    """Write ``cec2017_D<D>.npz`` for every supported dimension."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for dim in cec2017.SUPPORTED_DIMS:
        arrays: dict[str, np.ndarray] = {}
        for func_num in cec2017.FUNCTION_IDS:
            rotation, shift, shuffle = cec2017.read_official_files(input_dir, func_num, dim)
            arrays[f"M_{func_num}"] = rotation
            arrays[f"shift_{func_num}"] = shift
            if shuffle is not None:
                arrays[f"shuffle_{func_num}"] = shuffle.astype(np.int16)
        target = output_dir / f"cec2017_D{dim}.npz"
        np.savez_compressed(target, **arrays)
        written.append(target)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input-dir", type=Path, required=True, help="official input_data")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "cec2017",
    )
    args = parser.parse_args()
    for path in build(args.input_dir, args.output_dir):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{digest}  {path.name}  ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
