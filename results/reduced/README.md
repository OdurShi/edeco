# Example outputs of the reduced-scale configurations

Raw results, tables and figures written by the code of this repository for the
configurations listed below (laptop CPU, 4 worker processes, Python 3.11, NumPy 2.4,
SciPy 1.17).

| Directory | Command | Protocol |
|---|---|---|
| `cec2017_10d_eq11/` | `python scripts/run_experiment.py --config configs/reduced_cec2017_10d_eq11.yaml --output results/reduced/cec2017_10d_eq11` | CEC 2017, D = 10, 29 functions, ECO / EECO / DECO / EDECO with the default settings, N = 100, 50 000 evaluations per run, 10 runs |
| `cec2017_10d_eq19/` | `python scripts/run_experiment.py --config configs/reduced_cec2017_10d_eq19.yaml --output results/reduced/cec2017_10d_eq19` | same protocol and seeds with `dfs_student_rule: eq19` (Eq. 19 as printed) for DECO and EDECO |
| `engineering/` | `python scripts/run_experiment.py --config configs/reduced_engineering.yaml --output results/reduced/engineering` | RW01-RW10, ECO / EECO / DECO / EDECO, N = 100, 5000 D evaluations per run, 10 runs |
| `sensitivity_10d/` | `python scripts/run_experiment.py --config configs/sensitivity_10d.yaml --output results/reduced/sensitivity_10d` | 8 CEC 2017 functions, D = 10, 3 runs; one setting changed per entry |

Each directory contains

* `runs.csv`: one row per run (final value, error, evaluations, iterations, seed, time),
* `runs.jsonl`: the same runs with the best solution and the convergence history,
* `config.json`: resolved protocol, algorithm settings, library versions, wall time,
* `tables/`: Markdown and CSV tables written by `python scripts/make_tables.py --results <dir>`,
* `figures/`: PNG figures written by `python scripts/make_figures.py --results <dir>`.

The settings of these runs and a summary of their outcome are recorded in
`docs/implementation_notes.md`.
