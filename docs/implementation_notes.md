# Implementation notes

This document is the technical record of the repository: it maps the paper to the code,
documents every decision the paper leaves open, lists the places where the paper is
internally inconsistent, and records the runs carried out with this code.

Paper: W. Tang, S. Shi, Z. Lu, M. Lin, H. Cheng, "EDECO: An Enhanced Educational
Competition Optimizer for Numerical Optimization Problems", *Biomimetics* 2025, 10(3),
176, <https://doi.org/10.3390/biomimetics10030176>.

## 1. Provenance

* The paper's experiments were run in MATLAB R2023a (Section 4.1). The original EDECO
  source code was not available when this repository was written. Everything here is a
  **Python re-implementation written from the paper**.
* The base algorithm, ECO, follows the semantics of the reference MATLAB implementation
  published by its authors (Lian et al., 2024, `ECO.m`, <https://github.com/junbolian/ECO>),
  re-implemented in NumPy. No third-party source file is included.
* Random number streams differ from MATLAB, so individual runs cannot be matched; only
  statistics over runs are comparable.

## 2. Paper to code map

| Paper | Content | Code |
|---|---|---|
| Eq. (1)-(2) | logistic-map initialisation | `operators.logistic_map_init` |
| Eq. (3) | step weight `w` | `operators.step_weight` |
| Eq. (4) | primary stage, schools | `operators.primary_school_update` |
| Eq. (5) | primary stage, students | `operators.primary_student_update` |
| `close(X_i)` | nearest school | `operators.nearest_school` |
| `Levy(D)` | Lévy step | `operators.levy_step` |
| Eq. (6) | middle stage, schools | `operators.middle_school_update` |
| Eq. (7) | middle stage, students | `operators.middle_student_update` |
| Eq. (8) | amplitude `P` | `operators.amplitude` |
| Eq. (9) | talent coefficient `E` (as the product `P E`) | `operators.talent_product` |
| Eq. (10) | high stage, schools | `operators.high_school_update` |
| Eq. (11) | high stage, students | `operators.high_student_update` |
| Algorithm 1 | ECO main loop | `optimizer.minimize` with `ECOConfig()` |
| Eq. (12)-(14) | Gaussian model of the dominant group | `eda.fit_gaussian`, `eda.GaussianModel` |
| Eq. (15) | sampling | `eda.GaussianModel.sample` |
| Fig. 1, Sec. 3.1 | dominant group, greedy selection | `eda.select_dominant`, `eda.truncate_union` |
| Eq. (16)-(17) | DFS score | `dfs.dfs_scores`, `dfs.select_dfs` |
| Eq. (18) | DFS weight | `dfs.dfs_weight` |
| Eq. (19) | high stage, students with DFS | `operators.high_student_update` with `X_DFS` as leader (default reading), `operators.dfs_student_update` (as printed); see Section 6, item 1 |
| Algorithm 2, Fig. 2 | EDECO main loop | `optimizer.minimize` with `get_config("EDECO")` |
| Sec. 4.2 | ablation variants EECO, DECO | `optimizer.ALGORITHMS` (`use_eda`, `use_dfs`) |
| Table 1 | CEC 2017 suite | `benchmarks/cec2017.py` |
| Table 2 | parameter values | defaults of `optimizer.ECOConfig` |
| Sec. 4.1 | budget `5000 D`, 30 runs | `experiments.config.ExperimentConfig`, `configs/*.yaml` |
| Tables 3, 6; Figs. 6, 8 | Friedman ranking and test | `analysis.stats.friedman_test`, `analysis.tables.friedman_table`, `analysis.plots.friedman_figure` |
| Tables 4, 5; Fig. 5 | Wilcoxon rank-sum counts | `analysis.stats.rank_sum_counts`, `analysis.tables.rank_sum_table`, `analysis.plots.rank_sum_figure` |
| Eq. (20); Figs. 3, 7 | Nemenyi critical difference | `analysis.stats.nemenyi_critical_difference` |
| Tables A1-A4 | best / mean / std | `analysis.tables.summary_table` |
| Table 7 | engineering problems | `benchmarks/engineering.py` |
| Table 8 | engineering results | `analysis.tables.engineering_table` |

## 3. The algorithm as implemented

```
N individuals, budget Fes_max, g1 = round(0.2 N) primary schools, g2 = round(0.1 N) schools
N_d = round(0.5 N) dominant individuals, N_E = round(N / 3) EDA samples   (EECO / EDECO only)

initialise X by the logistic map, evaluate, sort by fitness
t_max = ceil((Fes_max - N) / (N + N_E))
for t = 1, 2, ... while evaluations remain:
    tau = t / t_max;  w = 0.1 ln(2 - tau);  P = 4 randn (1 - tau);  R ~ U(0, 1)
    c = P * E = pi * tau if R > H else P
    stage = t mod 3                                  (1 primary, 2 middle, 0 high)
    if stage == 0 and DFS: X_DFS = argmax_i S_i      (Eq. 16-18, current population)
    for i = 1 .. N:                                  (schools are the first g individuals)
        stage 1: school  X_i + w (mean(X_i) - X_i) Levy            student X_i + w (close - X_i) randn
        stage 2: school  X_i + (X_best - X_mean) e^(tau-1) Levy    student X_i - w close - (c w close - P X_i)
        stage 0: school  X_i + (X_best - X_i)(randn - randn')
                 student X_best - (c X_best - P X_i)                               (ECO, EECO)
                 student X_DFS  - (c X_best - P X_i)                               (DECO, EDECO)
        clamp to the bounds, evaluate, keep the candidate if it is not worse (greedy),
        update the global best X_best immediately
    if EDA: fit N(mu, C) to the N_d best, draw N_E samples, clamp, evaluate,
            keep the N best of the N + N_E individuals
    sort the population by fitness
```

`X_best` is the best solution found so far and is updated inside the sweep, exactly as
in the reference ECO code; `close`, `X_mean` and `X_DFS` are computed once per iteration
from the population at the start of the sweep.

The evaluation budget is enforced by a single wrapper (`evaluator.Evaluator`): a run
performs exactly `Fes_max` objective evaluations (the last iteration may be partial) and
records the best-so-far value after every evaluation.

## 4. Paper versus the reference ECO code

The paper restates ECO in Section 2. Where this restatement differs from the reference
code, the code semantics are the default (`base_equations="reference"`), because the
reference code defines the algorithm that ECO users actually run; the printed forms are
available with `base_equations="paper"`.

| Item | Paper | Reference ECO code | This repository |
|---|---|---|---|
| Eq. (1) initialisation | chain `x_i = 4 x_{i-1} (1 - x_{i-1})` | one logistic step from a fresh uniform number for every coordinate | reference |
| Eq. (4) `X_imean` | "average position of the vector for the i-th school agent" | `mean(X(j,:))`: scalar mean of the agent's own coordinates | same in both |
| Eq. (9) branch | `E = pi/P * t/t_max` if `R > H`, else 1 | uses the talent term if `R < H` | paper (equivalent for `H = 0.5`) |
| Eq. (9) `R` | one `R_t` per iteration | `R1` for the middle stage and `R2` for the high stage | one draw per iteration (only one stage is active) |
| Eq. (10) school, high stage | `X_i + (X_best - X_mean) randn - (X_best - X_mean) randn` | `X_i + (X_best - X_i) randn - (X_best - X_i) randn` | reference by default, paper form with `base_equations="paper"` |
| Eq. (11) student, high stage | `X_i - P (E X_best - X_i)` | `X_best - P (E X_best - X_i)` | reference by default, paper form with `base_equations="paper"` |
| `close(X_i)` | "the location of the school closest to X_i" | Manhattan distance, first school on ties | reference |
| Selection | not mentioned in Algorithm 1 | a candidate replaces its parent only if it is not worse; `X_best` updated at once; population sorted each iteration | reference |
| Stage test | `mod(t, 3) == 3` for the high stage | `else` branch (`mod(t, 3) == 0`) | reference (the printed condition can never be true) |
| End of run | - | `E = pi t / (P t_max)` divides by `P = 0` in the last iteration | the product `P E = pi t / t_max` is used, which is identical where `E` is defined |

## 5. Unspecified details and choices

The paper does not state the following; each choice is a parameter and can be changed.

| Detail | Choice (default) | Reason | Switch |
|---|---|---|---|
| Population size `N` | 100 | Not given anywhere in the paper. Chosen after the runs of Section 9.4: with `N = 30` and `N = 50` the EDA step concentrates the population early and EDECO ranks no better than ECO, with `N = 100` it ranks clearly better. | `pop_size` |
| Dominant group size `N_d` | `round(N / 2)` | "some individuals with better fitness values"; Fig. 1 draws 15 dominant out of 30 individuals; half of the population is the usual choice in EDA hybrids. | `dominant_ratio` |
| Number of EDA samples `N_E` | `round(N / 3)` | Only the symbol `N_E` appears (Section 3.4); Fig. 1 draws 10 samples for 30 individuals. | `eda_sample_ratio` |
| "Greedy strategy" after sampling | pool the `N` ECO individuals and the `N_E` samples, keep the `N` fittest | "Combining population and using greedy strategy" (Fig. 1), "select N individuals for the next iteration" (Section 3.3). | `eda_selection="pairwise"` lets every sample compete with one randomly chosen individual instead |
| Sampling from `N(mu, C)` when `C` is singular (`N_d <= D`) | `mu + A^T z / sqrt(N_d)` with `A` the centred dominant group | Exactly the distribution of Eq. (15) with the covariance of Eq. (14); needs no factorisation of a singular matrix. | - |
| Out-of-range EDA samples | clamped to the bounds | Same boundary control as ECO. | - |
| `norm(Fit_i)`, `norm(Dis_i)` in Eq. (16) | min-max normalisation; fitness oriented so that the best individual scores 1 | Convention of the fitness-distance-balance literature the strategy comes from. | - |
| Which individual is `X_DFS` | the one with the highest score; first index on ties | Fitness-distance-balance convention; deterministic tie-break. | - |
| Reference point of `Dis_i` | best individual of the current population | Eq. (17) uses `X_best`. | - |
| Eq. (18) | sawtooth `mod(t, t_max/alpha) / (t_max/alpha) * (1 - beta) + beta` | See Section 6, item 2. | `dfs_weight_schedule="literal"` |
| Form of the DFS student update (Eq. 19) | Eq. (11) with its leading best individual replaced by `X_DFS`: `X_DFS - P (E X_best - X_i)` | The paper is inconsistent; see Section 6, item 1, and the runs of Section 9.3. | `dfs_student_rule="eq19"` applies Eq. (19) as printed |
| `close(X_i)` inside Eq. (19) as printed | nearest of the `g2` high-stage schools | "The grouping of populations at the high school stage is the same as at the middle school stage" (Section 2.4). | - |
| `w` inside Eq. (19) as printed | the step weight of Eq. (3) | Eq. (19) is Eq. (7) with a new leading term; the DFS weight of Eq. (18) reuses the symbol but only enters the score. | - |
| Iteration horizon `t_max` under an evaluation budget | number of iterations that fit into `Fes_max`: `ceil((Fes_max - N) / (N + N_E))` | The paper stops on evaluations but the equations use `t / t_max`. | - |
| Lévy step | Mantegna's algorithm with index 1.5 | Reference ECO code. | `levy_beta` |
| Seeds | run `r` on function `f` uses `SeedSequence([seed, index of f, r])` for every algorithm | All algorithms start a run from the same initial population. | `seed` |
| Wilcoxon rank-sum test | `scipy.stats.mannwhitneyu`, two-sided, alpha = 0.05; direction from the rank sums | The paper names the test and the level only. | - |
| Friedman test | on the per-function mean results | Usual practice; the paper reports mean ranks and p-values. | - |
| Standard deviation in the tables | sample standard deviation (`ddof = 1`) | MATLAB's `std`. | - |
| Engineering formulations, penalty | see Section 8 | Not in the paper. | `penalty` |

## 6. Inconsistencies and probable misprints in the paper

1. **Eq. (19) and the stage it belongs to.** Section 3.2 says that DFS modifies "the
   update strategy at the high school level" and that `X_DFS` guides "the student
   population at the high school stage towards the optimal position"; the abstract
   describes the strategy as "replacing some of the best individual(s)"; Algorithm 2
   (line 21) and Fig. 2 use Eq. (19) in the high-school stage. The sentence that
   introduces Eq. (19), however, says "Equation (7) can be modified as follows", and the
   printed equation has the form of the *middle-school* rule Eq. (7) (with
   `w close(X_i)` terms and no best individual), not of the high-school rule Eq. (11).
   Two readings are implemented:
   * `dfs_student_rule="eq11"` (default): the high-school student rule Eq. (11), in the
     form of the reference ECO code, with its leading best individual replaced by
     `X_DFS`: `X_DFS - P (E X_best - X_i)`. This matches the abstract and the first
     paragraph of Section 3.2 and treats the body of Eq. (19) as copied from the wrong
     equation.
   * `dfs_student_rule="eq19"`: Eq. (19) exactly as printed, applied to the high-school
     students, as Algorithm 2 and Fig. 2 state.

   The default was chosen after running both (Section 9.3): with the first reading DFS
   improves on ECO and EDECO obtains the best Friedman rank of the four algorithms, as
   reported in the paper; with Eq. (19) as printed DECO is significantly worse than ECO
   on more functions than it is better. Which update the original MATLAB code uses can
   only be settled by the authors.
2. **Eq. (18).** As printed, `omega = mod(t_max/alpha, t) / t_max * (1 - beta) + beta`.
   For `t > t_max/alpha` the modulo equals `t_max/alpha`, so the weight is the constant
   `beta + (1 - beta)/alpha = 0.46`; for smaller `t` it lies between 0.40 and 0.46. A
   weight that barely moves contradicts the description of DFS as "dynamic" and
   "adaptive". In the dynamic fitness-distance balance method (dFDB, Kahraman et al.,
   2022) the distance weight is `0.6 * (1 - mod(t, T/f) / (T/f))`, i.e. the fitness
   weight is `0.4 + 0.6 * mod(t, T/f) / (T/f)` and sweeps its range `f` times per run
   (this is the formula of the MATLAB helper `dFDB.m` that circulates with FDB-based
   optimisers). With the arguments of `mod` swapped back and the period `t_max/alpha` as
   denominator, Eq. (18) is exactly this weight with `f = alpha = 10` and `beta = 0.4`.
   This is the default; the printed form is available as
   `dfs_weight_schedule="literal"`. In the small study of Section 9.4 (8 functions,
   3 runs) the printed form did not rank worse than the sawtooth (mean rank 2.38 against
   4.12), so the choice rests on the wording of the paper, not on those runs.
3. **Eq. (17)** has a comma instead of a plus sign after the first squared term; the
   Euclidean distance is meant.
4. **Eq. (10) and Eq. (11)** differ from the reference ECO code (Section 4).
5. **Algorithms 1 and 2** test `mod(t, 3) == 3` for the high-school stage, which is never
   true; `mod(t, 3) == 0` is meant. Algorithm 1 numbers two lines "24".
6. **Symbols.** `omega` denotes both the step weight of Eq. (3) and the DFS weight of
   Eq. (18); `alpha` denotes both the logistic-map constant (4) and the DFS parameter (10).
7. **Section 4.1** says the study uses "30-dimensional, 50-dimensional, and
   100-dimensional" CEC 2017 functions although 10-dimensional results are reported too.
8. **Population size, `N_d` and `N_E`** are not reported; Table 2 lists only
   `H, G1, G2, alpha, beta`.
9. **Eq. (20) and the critical difference values.** The text says the CDV "is obtained
   from Equation (4)" (Eq. 20 is meant) and that `q` comes from "the F distribution" (it
   is the studentised range statistic divided by sqrt(2)). With the stated values
   `q = 3.164`, `K = 10`, `N_f = 29`, Eq. (20) gives 2.5157, not the reported 2.3291; for
   the ablation study (`K = 4`, `q = 2.569`) it gives 0.8710, not the 0.80636 shown in
   Fig. 3. Both reported values are reproduced exactly when the denominator
   `6 N_f = 174` is replaced by 203 (= 7 x 29).
10. **Section 4.3** lists nine pairs of numbers for eight named competitors when
    summarising Table 5 (the ECO column is included but not named).
11. **Tables 4 and 5** report different EDECO-versus-ECO counts for the same dimension
    (10D: 28/1/0 and 27/2/0; 30D: 27/2/0 and 29/0/0; 50D: 27/2/0 and 26/3/0), presumably
    because the ablation and the comparison used separate sets of runs.
12. **Table 8** contains values that cannot be reached by feasible designs of the standard
    formulations (Section 8).
13. **ECO at D = 30.** In Tables A2-A4 the mean results of ECO at D = 30 are worse than at
    D = 50 on several functions and sometimes worse than at D = 100 (F1: 6.05E+09 at 30D,
    1.83E+08 at 50D, 1.62E+09 at 100D; F4: 1.04E+03, 6.76E+02, 1.16E+03; F12: 2.32E+08,
    2.72E+07, 2.38E+08), and the Friedman rank of ECO in Table 6 is 8.69 at 30D against
    6.45, 5.41 and 5.17 in the other dimensions. The 30D runs of ECO appear to have used
    a different setting from the other dimensions.
14. **Fig. 5** labels its bar groups "EDECO, ECO, SAO, CFOA, DBO, MRFO, ISGTOA, EMTLBO,
    TERIME", whereas the plotted counts are the rows "ECO, SAO, ..., AFDB-ARO" of
    Table 5 (the labels are shifted by one position).
15. **Smaller slips.** Section 4.3 refers to Figure 4 where Figure 5 is meant ("The
    winners and losers ... are visualized in Figure 4") and announces a "ranking" in
    Tables A1-A4, which contain best, mean and standard deviation only; Fig. 3 labels
    EECO as "EEDO" and Section 4.2 writes "EDCO" for DECO; the paragraph describing the
    primary school stage in Section 2 lacks its heading phrase; Table 8 prints the rank
    row of RW4 in exponential notation.

## 7. CEC 2017 suite

`benchmarks/cec2017.py` implements the official definitions of F1 and F3-F30 for
D = 10, 30, 50, 100 in NumPy. The shift vectors, rotation matrices and shuffle
permutations are shipped in `data/cec2017` as one NumPy archive per dimension, built
from the organisers' `input_data` files by `scripts/build_cec2017_data.py`; the archives
hold exactly the values obtained by parsing the text files (all 280 arrays were compared
with the text files). A directory with the official text files can be used instead
(`EDECO_CEC2017_DATA`), and the copy bundled with the optional `opfunu` package (its 328
files are identical to the organisers') serves as a fallback.

Validation performed for this repository:

* The organisers' C implementation (`cec17_func.cpp`, version of 26 May 2017) was
  compiled locally and evaluated at four deterministic probe points per function and
  dimension (two spread over the search range, two near the optimum): 464 values, stored
  in `tests/data/cec2017_reference.json`. The NumPy implementation agrees with all of
  them to a relative error below 2e-13 (`tests/test_benchmarks.py`, tolerance 1e-9).
  `scripts/generate_cec2017_reference.py` regenerates the file.
* `F_i(x*) = 100 i` holds at the stored optimum location for every function (Table 1).

Two properties of the official code are reproduced on purpose: F6 does not apply its
rotation matrix, and the rounding step of the "non-continuous" Rastrigin function F8 has
no effect. The Levy function F9 attains its optimum at `o + M^-1 1`, not at the shift
vector `o`; `CEC2017Function.x_opt` accounts for this.

Why the CEC 2017 classes of `opfunu` (version 1.0.4) are not used directly:

* opfunu numbers the 29 functions F1-F29: its F2-F29 are the official F3-F30, with the
  bias shifted by 100 (its F2 returns 200 at the optimum, the official F3 returns 300).
* Its F2-F19 load the shift, rotation and shuffle files of the official F2-F19 instead of
  F3-F20, so optimum locations and rotations differ from the official suite.
* Even with the data files re-mapped, its values agree with the official C code only for
  the official F1, F4 and F10 (relative difference about 1e-15). F12 agrees to about
  1e-7 and F18 to about 1e-4; F14 agrees to about 1e-4 at D = 30 and 50 but not at
  D = 10. For the other 23 functions the largest relative difference over 20 test points
  per function exceeds 0.2 (measured at D = 10, 30 and 50).

## 8. Engineering design problems

### 8.1 Formulations and penalty

The paper names ten problems (Table 7) and says that "the penalty function method was
used"; it gives no formulation, no penalty coefficient and no feasibility criterion.
`benchmarks/engineering.py` therefore implements the standard formulations of the
literature. Each one was checked by evaluating a published or best-known solution
(`reference_x`): it is feasible and reproduces the published optimum
(`tests/test_benchmarks.py`). For RW01, RW03-RW05, RW07, RW08 and RW10 the optimum was
additionally re-derived with SciPy's SLSQP (multi-start; one continuous problem per
integer value for RW07) and agrees with the published value; the optimum of RW02 follows
in closed form from its three active constraints and the bound `L = 200`.

| Key | Problem | D | Constraints | Best-known feasible value used as reference | Source of the formulation |
|---|---|---|---|---|---|
| RW01 | Tension/compression spring | 3 | 4 | 0.012665233 | Arora (1989); Coello & Mezura-Montes (2002) |
| RW02 | Pressure vessel, continuous variables | 4 | 4 | 5885.3327736 | Sandgren (1990); Kumar et al. (2020) |
| RW03 | Three-bar truss | 2 | 3 | 263.895843 | Ray & Saini (2001) |
| RW04 | Welded beam (seven constraints) | 4 | 7 | 1.724852309 | Ragsdell & Phillips (1976); Coello (2000) |
| RW05 | Speed reducer (integer number of teeth) | 7 | 11 | 2994.471066 | Golinski (1973); Mezura-Montes & Coello (2005) |
| RW06 | Gear train (integer teeth numbers) | 4 | 0 | 2.700857e-12 | Sandgren (1990) |
| RW07 | Rolling element bearing (integer number of balls) | 10 | 9 | -81859.74 (capacity is maximised) | Gupta et al. (2007); Rao et al. (2011) |
| RW08 | Cantilever beam | 5 | 1 | 1.339956 | Chickermane & Gea (1996) |
| RW09 | Multiple disk clutch brake (discrete variables) | 5 | 8 | 0.313656611 | Osyczka (2002); Deb & Srinivasan (2006) |
| RW10 | Step-cone pulley | 5 | 8 + 3 equalities | 16.069868725 | Rao (2009); Kumar et al. (2020) |

Constraint handling: static quadratic penalty
`F(x) = f(x) + rho * sum_i max(0, g_i(x))^2` with `rho = 1e12` (`DEFAULT_PENALTY`).
Equalities `h(x) = 0` are relaxed to `|h(x)| <= 1e-4`. Discrete variables are rounded to
the nearest admissible value before `f` and `g` are evaluated (`repair`). A returned
solution is reported as feasible when it respects the bounds and
`max_i g_i(x) <= 1e-6`. The coefficient is deliberately large: with a small coefficient
the minimiser of the penalised function lies outside the feasible region and has a lower
objective value than any feasible design. The engineering table reports the design
objective `f` over the runs whose returned solution is feasible, together with the number
of such runs, and ranks the algorithms by the mean penalised objective.

Unit handling in RW09: radii are in millimetres, so the friction torque is converted to
N m and the sliding velocity to m/s before they are compared with `M_s = 40 N m`,
`M_f = 3 N m` and `v_sr,max = 10 m/s`. With these conversions the published optimum
(`r_i = 70, r_o = 90, t = 1, Z = 3`, mass 0.313657) is feasible and optimal.

### 8.2 Comparison with the values reported in the paper's Table 8

The lowest "Best" value over the ten algorithms of the paper's Table 8 is compared with
the best-known feasible optimum of the standard formulation:

| Problem | Lowest best value in the paper's Table 8 | Best-known feasible optimum | Remark |
|---|---|---|---|
| RW01 | 1.2666E-02 | 0.012665233 | consistent |
| RW02 | 5.8701E+03 | 5885.3327736 | 0.26 % below the feasible optimum |
| RW03 | 2.6389E+02 | 263.895843 | consistent |
| RW04 | 1.6928E+00 | 1.724852309 | 1.9 % below the optimum of the seven-constraint formulation |
| RW05 | 2.9936E+03 | 2994.471066 | 0.03 % below the feasible optimum |
| RW06 | 2.7009E-12 | 2.700857e-12 | consistent |
| RW07 | -2.4358E+05 | -81859.74 | about three times the best-known load capacity |
| RW08 | 1.3400E+00 | 1.339956 | consistent |
| RW09 | 3.9247E+08 (all ten algorithms, Std 1.8187E-07) | 0.313656611 | nine orders of magnitude larger |
| RW10 | 1.6086E+01 | 16.069868725 | consistent (0.1 % above) |

Observations:

* **RW09.** Every algorithm reports the same value 3.9247E+08 with a standard deviation
  of 1.8E-07. A mass of a few hundred grams is expected. The value is reproduced to five
  digits by `100 * (g4^2 + g5^2)` at `r_i = 60, r_o = 90, F = 600`, where
  `g5 = v_sr - v_sr,max` and `g4 = p_rz v_sr - p_max v_sr,max` are evaluated with the
  sliding velocity in mm/s (1989.7) against the limit 10 (m/s): 100 * (74.44^2 +
  1979.67^2) = 3.92466E+08. This suggests that, in the code behind Table 8, the velocity
  constraints can never be satisfied (unit mismatch), every algorithm merely minimises a
  penalty term with coefficient 100, and the reported number is that penalty rather than
  a design objective; the row would then carry no ranking information.
* **RW07.** A dynamic load capacity of 243 580 is far above the best-known value
  (81 859.74 with the formulation used here; 85 549.24 is also widely quoted). With the
  formulation used here such values are only reached when the assembly-angle constraint
  that limits the number of balls is ignored: without it, `Z = 50` balls give 233 438.
  If, in addition, the exponent `10/3` in the factor `f_c` is applied as `^10/3` (a
  common transcription of the formula, which also yields values near 85 500 for
  `Z = 11`), the maximum is 243 355, within 0.1 % of the paper's value. The designs
  behind this row therefore most likely violate that constraint.
* **RW02, RW04, RW05.** The reported best values lie below the best-known feasible
  optima of the standard formulations. Either different formulations were used, or the
  penalty coefficient was small enough for slightly infeasible designs to be returned.
  The paper does not contain the information needed to decide.
* The rank row of RW4 is printed in exponential notation (`2.00E+00`, ...), and the rank
  row of RW9 shows rank 1 for all algorithms although the Friedman ranking in the last
  row of the table is consistent with the tied rank 5.5.

## 9. Record of runs

Everything in this section was obtained with the code of this repository on a laptop CPU
(Apple M-series, 4 worker processes, Python 3.11, NumPy 2.4, SciPy 1.17). The raw
outputs are in `results/reduced/`.

### 9.1 Test suite

`pytest`: 156 tests pass (unit tests of every operator, the Gaussian model, the
DFS score, evaluation accounting and seeding, the benchmarks and data files, the
statistics, and end-to-end runs of the experiment runner and the command line).
`ruff check .` reports no findings.

### 9.2 Benchmark checks

* CEC 2017: agreement with the organisers' C code at 464 probe points (Section 7).
* Engineering problems: feasibility and objective value of the reference solutions and
  re-derivation of the optima with SciPy (Section 8.1).

### 9.3 Ablation study at D = 10 with 10 runs

Configurations `configs/reduced_cec2017_10d_eq11.yaml` (default reading of the DFS
update) and `configs/reduced_cec2017_10d_eq19.yaml` (Eq. 19 as printed): 29 functions,
D = 10, `N = 100`, 50 000 evaluations per run, 10 runs per function and algorithm, same
seeds in both experiments. ECO and EECO do not use DFS and give identical results in
both. Raw results, tables and figures: `results/reduced/cec2017_10d_eq11` and
`results/reduced/cec2017_10d_eq19`.

Friedman mean ranks over the 29 functions:

| DFS update | ECO | EECO | DECO | EDECO | p-value |
|---|---|---|---|---|---|
| `eq11` (default) | 3.45 | 2.28 | 2.59 | 1.69 | 3.50E-06 |
| `eq19` (as printed) | 3.10 | 1.93 | 2.83 | 2.14 | 1.07E-03 |

Wilcoxon rank-sum test at the 0.05 level, `+/=/-` = number of functions on which the
first algorithm is significantly better / not significantly different / significantly
worse, and number of functions on which the variant has a lower mean error than ECO:

| DFS update | EECO vs. ECO | DECO vs. ECO | EDECO vs. ECO | EDECO vs. EECO | EDECO vs. DECO | lower mean than ECO (EECO / DECO / EDECO) |
|---|---|---|---|---|---|---|
| `eq11` (default) | 10/17/2 | 6/21/2 | 14/14/1 | 2/27/0 | 11/17/1 | 24 / 23 / 24 |
| `eq19` (as printed) | 10/17/2 | 2/19/8 | 11/15/3 | 2/26/1 | 13/16/0 | 24 / 19 / 18 |

Mean error `f - f*` over the 10 runs. The last two columns give, for orientation, the
values printed in Table A1 of the paper (mean of 30 runs minus `100 i`, MATLAB code,
population size not stated):

| Function | ECO | EECO | DECO (`eq11`) | EDECO (`eq11`) | DECO (`eq19`) | EDECO (`eq19`) | ECO (paper) | EDECO (paper) |
|---|---|---|---|---|---|---|---|---|
| F1 | 5.211E+03 | 3.431E+03 | 2.830E+03 | 2.364E+03 | 1.633E+05 | 2.026E+03 | 1.060E+04 | 3.296E+02 |
| F3 | 5.708E-02 | 1.484E+01 | 2.293E-01 | 8.649E-01 | 2.776E+01 | 5.556E+00 | 2.631E+01 | < 0.01 |
| F4 | 1.898E+01 | 4.287E+00 | 5.750E+00 | 4.837E+00 | 5.375E+00 | 4.226E+00 | 6.680E+00 | 3.020E+00 |
| F5 | 2.339E+01 | 9.952E+00 | 1.660E+01 | 8.514E+00 | 1.404E+01 | 1.170E+01 | 2.259E+01 | 8.860E+00 |
| F6 | 2.776E+00 | 1.493E+00 | 1.800E+00 | 1.039E+00 | 4.083E+00 | 4.424E-01 | 6.010E+00 | 2.000E-02 |
| F7 | 3.993E+01 | 2.176E+01 | 3.955E+01 | 1.969E+01 | 3.465E+01 | 2.013E+01 | 3.383E+01 | 1.959E+01 |
| F8 | 1.841E+01 | 1.015E+01 | 1.564E+01 | 1.016E+01 | 1.618E+01 | 9.662E+00 | 1.797E+01 | 1.035E+01 |
| F9 | 1.727E+01 | 1.146E+01 | 5.782E-01 | 9.501E-01 | 6.371E+00 | 5.578E+00 | 4.177E+01 | 2.000E-02 |
| F10 | 8.247E+02 | 6.278E+02 | 5.676E+02 | 5.405E+02 | 6.885E+02 | 3.475E+02 | 6.756E+02 | 5.839E+02 |
| F11 | 5.222E+01 | 2.118E+01 | 2.305E+01 | 1.707E+01 | 2.564E+01 | 3.003E+01 | 4.630E+01 | 4.900E+00 |
| F12 | 1.875E+04 | 6.541E+03 | 2.328E+04 | 6.165E+03 | 6.748E+05 | 2.431E+05 | 8.216E+04 | 7.576E+02 |
| F13 | 1.623E+03 | 6.579E+02 | 1.375E+03 | 1.024E+03 | 4.794E+03 | 1.090E+03 | 1.005E+03 | 4.780E+01 |
| F14 | 8.598E+01 | 3.032E+01 | 5.263E+01 | 2.778E+01 | 8.382E+01 | 2.889E+01 | 7.690E+01 | 1.680E+01 |
| F15 | 9.311E+01 | 4.770E+01 | 9.007E+01 | 3.982E+01 | 4.079E+02 | 2.154E+02 | 1.551E+02 | 8.000E+00 |
| F16 | 1.155E+02 | 3.746E+01 | 5.427E+01 | 1.955E+01 | 5.418E+01 | 1.798E+01 | 8.320E+01 | 1.380E+01 |
| F17 | 5.915E+01 | 3.406E+01 | 4.492E+01 | 3.041E+01 | 5.508E+01 | 3.540E+01 | 5.290E+01 | 4.180E+01 |
| F18 | 3.242E+03 | 2.122E+02 | 3.908E+03 | 1.918E+02 | 1.111E+04 | 1.924E+03 | 3.373E+03 | 3.840E+01 |
| F19 | 1.956E+02 | 1.596E+01 | 2.976E+01 | 3.180E+01 | 5.113E+02 | 3.659E+02 | 5.160E+01 | 4.100E+00 |
| F20 | 5.737E+01 | 3.903E+01 | 5.675E+01 | 3.067E+01 | 5.465E+01 | 2.447E+01 | 6.750E+01 | 4.270E+01 |
| F21 | 1.274E+02 | 1.248E+02 | 1.585E+02 | 1.466E+02 | 1.504E+02 | 1.364E+02 | 1.122E+02 | 1.009E+02 |
| F22 | 8.585E+01 | 8.112E+01 | 7.264E+01 | 8.305E+01 | 8.365E+01 | 9.867E+01 | 5.920E+01 | 4.840E+01 |
| F23 | 3.290E+02 | 3.214E+02 | 3.282E+02 | 3.181E+02 | 3.218E+02 | 3.113E+02 | 3.256E+02 | 3.135E+02 |
| F24 | 3.043E+02 | 3.204E+02 | 2.733E+02 | 3.181E+02 | 2.771E+02 | 3.426E+02 | 2.684E+02 | 2.513E+02 |
| F25 | 4.270E+02 | 4.309E+02 | 4.318E+02 | 4.178E+02 | 4.224E+02 | 4.449E+02 | 4.242E+02 | 4.212E+02 |
| F26 | 3.838E+02 | 3.609E+02 | 3.340E+02 | 3.735E+02 | 3.589E+02 | 4.081E+02 | 3.496E+02 | 3.000E+02 |
| F27 | 3.941E+02 | 4.163E+02 | 3.925E+02 | 4.067E+02 | 3.929E+02 | 3.959E+02 | 3.938E+02 | 3.914E+02 |
| F28 | 5.968E+02 | 5.929E+02 | 4.209E+02 | 5.304E+02 | 4.472E+02 | 5.493E+02 | 5.182E+02 | 4.417E+02 |
| F29 | 3.128E+02 | 2.874E+02 | 2.945E+02 | 2.534E+02 | 2.923E+02 | 2.636E+02 | 3.046E+02 | 2.613E+02 |
| F30 | 4.453E+03 | 1.712E+05 | 6.507E+04 | 4.329E+04 | 1.563E+05 | 2.464E+05 | 1.480E+05 | 1.405E+04 |

Wall-clock time with 4 worker processes on a laptop CPU: 637 s (`eq11`) and 796 s (`eq19`) for 1160 runs each.

### 9.4 Sensitivity to settings the paper leaves open

Configuration `configs/sensitivity_10d.yaml`: 8 functions
(F1, F3, F5, F6, F9, F10, F21, F23), D = 10, 50 000 evaluations, 3 runs; every entry changes one
setting of the default ECO, DECO or EDECO configuration. Mean error over the runs and
mean rank over the functions among the 14 entries (results in
`results/reduced/sensitivity_10d`; with 3 runs the figures indicate tendencies only):

| Entry | Changed setting | Mean rank | F1 | F3 | F5 | F6 | F9 | F10 | F21 | F23 |
|---|---|---|---|---|---|---|---|---|---|---|
| `ECO` | - | 7.00 | 1.51E+03 | 8.99E-02 | 1.53E+01 | 1.06E+01 | 1.15E+00 | 6.90E+02 | 1.04E+02 | 3.30E+02 |
| `EDECO` | - | 4.12 | 1.99E+03 | 9.09E-02 | 9.30E+00 | 3.07E+00 | 1.52E-01 | 5.55E+02 | 1.03E+02 | 3.09E+02 |
| `ECO-N30` | `pop_size = 30` | 11.50 | 2.03E+03 | 3.64E-01 | 2.06E+01 | 1.12E+01 | 1.92E+02 | 7.64E+02 | 1.94E+02 | 3.33E+02 |
| `EDECO-N30` | `pop_size = 30` | 12.50 | 3.77E+04 | 7.05E+00 | 2.62E+01 | 1.45E+01 | 1.74E+02 | 6.10E+02 | 1.48E+02 | 3.48E+02 |
| `ECO-N50` | `pop_size = 50` | 10.25 | 8.47E+03 | 1.15E+00 | 2.02E+01 | 8.06E+00 | 1.69E+02 | 5.54E+02 | 1.52E+02 | 3.29E+02 |
| `EDECO-N50` | `pop_size = 50` | 8.50 | 1.32E+04 | 1.00E+00 | 1.06E+01 | 7.75E+00 | 4.54E+01 | 6.08E+02 | 1.39E+02 | 3.17E+02 |
| `ECO-paper-eqs` | `base_equations = paper` | 8.88 | 4.26E+03 | 2.08E-01 | 2.69E+01 | 1.45E+00 | 1.14E+01 | 7.24E+02 | 1.40E+02 | 3.24E+02 |
| `DECO` | - | 7.88 | 2.16E+03 | 4.94E-01 | 1.64E+01 | 7.02E-01 | 2.50E-02 | 7.94E+02 | 1.43E+02 | 3.20E+02 |
| `DECO-eq19` | `dfs_student_rule = eq19` | 8.50 | 1.39E+05 | 2.79E+01 | 1.28E+01 | 1.38E+00 | 2.47E+01 | 6.05E+02 | 1.04E+02 | 3.26E+02 |
| `EDECO-eq19` | `dfs_student_rule = eq19` | 4.25 | 1.77E+03 | 1.56E-01 | 1.08E+01 | 5.17E-02 | 3.85E+00 | 4.61E+02 | 1.40E+02 | 3.12E+02 |
| `EDECO-literal-w` | `dfs_weight_schedule = literal` | 2.38 | 1.22E+03 | 3.27E-02 | 8.96E+00 | 6.69E-02 | 3.33E-01 | 5.11E+02 | 1.37E+02 | 3.09E+02 |
| `EDECO-pairwise` | `eda_selection = pairwise` | 4.75 | 9.10E+02 | 5.61E-02 | 8.95E+00 | 1.96E-01 | 1.47E+00 | 6.02E+02 | 1.75E+02 | 3.11E+02 |
| `EDECO-NE-N` | `eda_sample_ratio = 1` | 7.50 | 8.57E+03 | 4.45E-01 | 1.49E+01 | 6.27E-02 | 6.22E+01 | 5.34E+02 | 1.74E+02 | 3.09E+02 |
| `EDECO-Nd-0.3N` | `dominant_ratio = 0.3` | 7.00 | 3.72E+03 | 2.13E-01 | 6.97E+00 | 2.29E+00 | 4.16E+01 | 5.31E+02 | 1.40E+02 | 3.33E+02 |

Wall-clock time: 93 s for 336 runs.

### 9.5 Engineering design problems with 10 runs

Configuration `configs/reduced_engineering.yaml`: RW01-RW10, `N = 100`, `5000 D`
evaluations per run, 10 runs, penalty coefficient `1e12` (results in
`results/reduced/engineering`). Mean design objective over the runs whose returned
solution is feasible, with the number of feasible runs in parentheses; the last column is
the best-known feasible value of Section 8.1:

| Problem | ECO | EECO | DECO | EDECO | Reference |
|---|---|---|---|---|---|
| RW01 | 0.0129147 (10/10) | 0.0131775 (10/10) | 0.013289 (10/10) | 0.0133634 (10/10) | 0.012665233 |
| RW02 | 6799.09 (10/10) | 6380 (10/10) | 6812.52 (10/10) | 6214.9 (10/10) | 5885.33277 |
| RW03 | 263.908 (10/10) | 263.899 (10/10) | 263.9 (10/10) | 263.899 (10/10) | 263.895843 |
| RW04 | 1.97899 (10/10) | 1.85275 (10/10) | 1.96545 (10/10) | 1.80932 (10/10) | 1.72485231 |
| RW05 | 3003.4 (10/10) | 2994.48 (10/10) | 2996.6 (10/10) | 2994.48 (10/10) | 2994.47107 |
| RW06 | 9.31659E-10 (10/10) | 2.25593E-11 (10/10) | 1.24119E-09 (10/10) | 2.42806E-11 (10/10) | 2.700857E-12 |
| RW07 | -81433.4 (10/10) | -78685 (10/10) | -81853.8 (10/10) | -81848.6 (10/10) | -81859.74 |
| RW08 | 1.34389 (10/10) | 1.34108 (10/10) | 1.35767 (10/10) | 1.34431 (10/10) | 1.339956 |
| RW09 | 0.324194 (10/10) | 0.321559 (10/10) | 0.321559 (10/10) | 0.316291 (10/10) | 0.313656611 |
| RW10 | 16.9931 (10/10) | 16.772 (10/10) | 17.0911 (10/10) | 16.7977 (10/10) | 16.0698687 |
| Friedman mean rank (penalised objective) | 3.10 | 1.85 | 3.15 | 1.90 | |

Best feasible objective found by any algorithm in any run: RW01 0.01266565, RW02 5885.532, RW03 263.8958, RW04 1.725461, RW05 2994.471, RW06 2.700857E-12, RW07 -81859.74, RW08 1.339984, RW09 0.3136566, RW10 16.25241.

Wall-clock time: 38 s for 400 runs.

### 9.6 Single runs at D = 30, 50 and 100

One run per entry with the default settings (`N = 100`, `Fes_max = 5000 D`, seed 1),
to exercise the code at the higher dimensions of the paper and to time it (single
process):

| D | Function | Algorithm | Error `f - f*` | Evaluations | Iterations | Time |
|---|---|---|---|---|---|---|
| 30 | F1 | ECO | 1.309E+05 | 150000 | 1499 | 0.7s |
| 30 | F1 | EDECO | 4.925E+05 | 150000 | 1128 | 0.7s |
| 30 | F5 | ECO | 1.105E+02 | 150000 | 1499 | 0.9s |
| 30 | F5 | EDECO | 1.304E+02 | 150000 | 1128 | 0.9s |
| 30 | F21 | ECO | 3.245E+02 | 150000 | 1499 | 3.8s |
| 30 | F21 | EDECO | 2.925E+02 | 150000 | 1128 | 3.5s |
| 50 | F5 | ECO | 3.206E+02 | 250000 | 2499 | 1.8s |
| 50 | F5 | EDECO | 2.094E+02 | 250000 | 1879 | 1.8s |
| 100 | F5 | ECO | 7.373E+02 | 500000 | 4999 | 4.1s |
| 100 | F5 | EDECO | 6.689E+02 | 500000 | 3759 | 3.8s |

Sections 9.3-9.6 list all optimisation runs carried out for this repository. The
configurations with the paper's protocol of 30 runs (`configs/cec2017_*.yaml`,
`configs/engineering.yaml`) are provided for the reader to run.

## 10. Scope

* **Algorithms.** ECO, EECO, DECO and EDECO are implemented. The eight comparison
  algorithms of Section 4.3 (SAO, CFOA, DBO, MRFO, ISGTOA, EMTLBO, TERIME, AFDB-ARO) are
  not part of this repository, so the tables and figures that involve them (Tables 5 and
  6, Figs. 4-7, the competitor columns of Tables A1-A4 and 8) are outside its scope; the
  ECO, EECO, DECO and EDECO parts of the evaluation are covered.
* **Runs.** Configurations with the paper's protocol (30 runs, D = 10, 30, 50, 100) are
  provided in `configs/`. The runs carried out for this repository are those listed in
  Section 9.
* **Reference implementation.** The base ECO was written from the reference MATLAB file
  `ECO.m`; MATLAB was not available, so no MATLAB run was compared with the Python code.
  Random number streams differ from MATLAB in any case.
* **Settings not given in the paper.** Population size, `N_d`, `N_E`, the form of the
  greedy selection, the engineering formulations and the penalty coefficient are choices
  of this repository (Sections 5 and 8); absolute numbers therefore depend on them.
* **Figures.** `edeco figures` draws convergence curves, box plots, the Friedman ranking
  with the critical difference value and the rank-sum counts. Radar charts (Fig. 4) and
  critical-difference diagrams (Figs. 3 and 7) are not drawn.
* **CEC 2017 dimensions.** D = 10, 30, 50 and 100 (the dimensions of the paper).
