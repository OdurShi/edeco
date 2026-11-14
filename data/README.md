# Data

Everything the experiments need is in this directory; nothing is downloaded at run time.
The paper uses no external dataset: its "data" are the definition files of the CEC 2017
benchmark suite and the formulations of ten engineering design problems.

## CEC 2017 benchmark definitions (`cec2017/`)

* **What.** Shift vectors, rotation matrices and shuffle permutations that define the
  29 functions F1 and F3-F30 of the CEC 2017 single-objective bound-constrained suite
  (F2 was removed from the suite by its organisers) for the dimensions used in the paper,
  D = 10, 30, 50 and 100. The function formulas themselves are implemented in
  `src/edeco/benchmarks/cec2017.py`.
* **Used by.** The CEC 2017 experiments: `configs/cec2017_*.yaml`,
  `configs/reduced_cec2017_*.yaml`, `configs/sensitivity_10d.yaml` (Tables 3-6 and A1-A4
  of the paper).
* **Source.** The `input_data` directory of the organisers' distribution of the test
  suite: N. H. Awad, M. Z. Ali, J. J. Liang, B. Y. Qu, P. N. Suganthan, "Problem
  Definitions and Evaluation Criteria for the CEC 2017 Special Session and Competition on
  Single Objective Real-Parameter Numerical Optimization", Technical Report, Nanyang
  Technological University, Singapore, 2016;
  <https://github.com/P-N-Suganthan/CEC2017-BoundContrained>.
* **Version.** Data files accompanying `cec17_func.cpp`, "Last update on 26_5_2017".
* **Licence.** The organisers publish the files for use with the benchmark; their
  distribution contains no explicit licence text. The values are redistributed here
  unchanged, with attribution to the authors above.
* **Size.** 29 functions x 4 dimensions. Per dimension: 29 rotation arrays, 29 shift
  arrays and 12 shuffle arrays (hybrid functions F11-F20 and compositions F29-F30).
* **Layout.** One compressed NumPy archive per dimension, `cec2017_D<D>.npz`, with the
  arrays
  * `M_<f>`: rotation matrices, shape `(parts, D, D)`, float64;
  * `shift_<f>`: shift vectors, shape `(parts, D)`, float64;
  * `shuffle_<f>`: shuffle permutations with 1-based indices as in the official files,
    shape `(parts, D)`, int16;

  where `<f>` is the official function number and `parts` is 10 for the composition
  functions F21-F30 and 1 otherwise. The arrays hold exactly the 64-bit values obtained
  by parsing the official text files (`M_<f>_D<D>.txt`, `shift_data_<f>.txt`,
  `shuffle_data_<f>_D<D>.txt`); `scripts/build_cec2017_data.py --input-dir <input_data>`
  rebuilds the archives from those files.
* **Loading.** `edeco.benchmarks.cec2017` reads this directory by default. Another
  directory (holding these archives or the official text files) can be selected with the
  `--data-dir` option of `edeco solve` and `edeco run`, the `EDECO_CEC2017_DATA`
  environment variable or the `data_dir` argument of the Python API.

## Engineering design problems (`engineering/`)

* **What.** `problems.json` summarises the ten constrained design problems RW01-RW10
  (Table 7 of the paper) as implemented in `src/edeco/benchmarks/engineering.py`: name,
  dimension, variable bounds, number of constraints, discrete variables, a reference
  solution with its objective value, the literature source of each formulation, and the
  constraint-handling settings.
* **Used by.** `configs/engineering.yaml`, `configs/reduced_engineering.yaml` (Table 8 of
  the paper).
* **Source.** Written by `scripts/export_engineering_problems.py` from the code; the
  formulations come from the publications named in the `source` field of each problem.
* **Licence.** MIT, as the rest of this repository.
* **Size.** 10 problems.

## Files and checksums

| File | Size | SHA-256 |
|---|---|---|
| `cec2017/cec2017_D10.npz` | 75 KiB | `b14d3de88fdd8cd1a93a094fbf382ebfef7d99b7000a3a9f020f1bcf9c79f90d` |
| `cec2017/cec2017_D30.npz` | 309 KiB | `c3e7fb179348b300c80d7eb187ad5ebf349b42d9cdf81d30341a2af1bde1a8b8` |
| `cec2017/cec2017_D50.npz` | 666 KiB | `96cc54e67da476d72d07aeca0e87571d3d4a7460827315f6dc34180c7cf30e99` |
| `cec2017/cec2017_D100.npz` | 1403 KiB | `22ca931569edeb54d578c7e766177c148f927aff79d43585bb5a4c6ca2f9cb31` |
| `engineering/problems.json` | 7 KiB | `bd1e92a0189acc84903f37e6d35a3a6b1e195c81118a797c0363073b3a126519` |

The same checksums are in `SHA256SUMS` (`shasum -a 256 -c SHA256SUMS` from this
directory).
