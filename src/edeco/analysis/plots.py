"""Figures computed from the raw results of an experiment.

* ``convergence_figure``  mean best-so-far error against function evaluations,
* ``boxplot_figure``      distribution of the final results over the independent runs,
* ``friedman_figure``     Friedman mean ranks with the Nemenyi critical difference
  (the information of Figs. 3, 6, 7 and 8 of the paper),
* ``rank_sum_figure``     Wilcoxon rank-sum ``+ / = / -`` counts (Fig. 5 of the paper).

The paper itself shows no convergence curves or box plots; they are provided because
they are the customary way to inspect such experiments.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from edeco.analysis import stats

logger = logging.getLogger(__name__)

# Categorical slots, assigned to the algorithms in a fixed order (never cycled).
SERIES_COLORS = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
LINE_STYLES = ("-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (6, 2, 2, 2)))
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
OUTCOME_COLORS = {"+": "#2a78d6", "=": "#b9b8b2", "-": "#eb6834"}
OUTCOME_LABELS = {
    "+": "significantly better",
    "=": "no significant difference",
    "-": "significantly worse",
}


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(which="both", colors=TEXT_SECONDARY, labelsize=7, length=2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _color(index: int) -> str:
    if index >= len(SERIES_COLORS):
        raise ValueError(f"at most {len(SERIES_COLORS)} algorithms can be drawn in one figure")
    return SERIES_COLORS[index]


def _grid(n_panels: int, columns: int) -> tuple[Figure, np.ndarray]:
    rows = math.ceil(n_panels / columns)
    fig, axes = plt.subplots(
        rows, columns, figsize=(2.6 * columns, 2.1 * rows + 0.6), squeeze=False
    )
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.ravel()[n_panels:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def _positive(values: np.ndarray) -> np.ndarray:
    """Clip to a small positive floor so that exact zeros can be drawn on a log axis."""
    return np.maximum(values, 1e-12)


def convergence_figure(results_dir: str | Path, columns: int = 5) -> Figure:
    """Mean best-so-far value over the runs against the number of evaluations."""
    records = stats.load_records(results_dir)
    results = stats.load_results(results_dir)
    use_error = results.metric == "error"
    curves: dict[tuple[str, str], list[list[float]]] = {}
    evals: dict[str, list[int]] = {}
    offsets: dict[str, float] = {}
    for record in records:
        curves.setdefault((record["function"], record["algorithm"]), []).append(record["history"])
        evals[record["function"]] = record["history_evals"]
        offsets[record["function"]] = record["best_f"] - record["error"] if use_error else 0.0

    fig, axes = _grid(len(results.functions), columns)
    for ax, function in zip(axes, results.functions, strict=False):
        _style(ax)
        panel = [
            np.mean(np.array(curves[(function, a)]), axis=0) - offsets[function]
            for a in results.algorithms
        ]
        log_scale = use_error or all(np.all(curve > 0.0) for curve in panel)
        for i, (algorithm, mean_curve) in enumerate(zip(results.algorithms, panel, strict=True)):
            ax.plot(
                evals[function],
                _positive(mean_curve) if log_scale else mean_curve,
                color=_color(i),
                linestyle=LINE_STYLES[i],
                linewidth=1.6,
                label=algorithm,
            )
        # Objectives that take negative values (maximisation problems written as
        # minimisation) cannot use a log axis; a symmetric log axis keeps the large
        # penalised values of the first evaluations from flattening the curve.
        ax.set_yscale("log" if log_scale else "symlog")
        ax.set_title(function, fontsize=8, color=TEXT_PRIMARY, loc="left")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
        ax.xaxis.get_offset_text().set_size(6)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False, fontsize=8)
    ylabel = "mean error f - f*" if use_error else "mean best objective"
    fig.supxlabel("function evaluations", fontsize=8, color=TEXT_SECONDARY)
    fig.supylabel(ylabel, fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 0.96))
    return fig


def boxplot_figure(results_dir: str | Path, columns: int = 5) -> Figure:
    """Box plots of the final results of the independent runs, one panel per function."""
    results = stats.load_results(results_dir)
    fig, axes = _grid(len(results.functions), columns)
    for ax, function in zip(axes, results.functions, strict=False):
        _style(ax)
        data = [results.values[function][a] for a in results.algorithms]
        positive = all(np.all(d > 0.0) for d in data)
        boxes = ax.boxplot(
            data,
            patch_artist=True,
            widths=0.5,
            medianprops={"color": TEXT_PRIMARY, "linewidth": 1.2},
            whiskerprops={"color": TEXT_SECONDARY, "linewidth": 0.8},
            capprops={"color": TEXT_SECONDARY, "linewidth": 0.8},
            flierprops={
                "marker": "o",
                "markersize": 2.5,
                "markerfacecolor": TEXT_SECONDARY,
                "markeredgecolor": "none",
            },
        )
        for i, patch in enumerate(boxes["boxes"]):
            patch.set_facecolor(_color(i))
            patch.set_edgecolor(SURFACE)
            patch.set_alpha(0.9)
        ax.set_xticks(range(1, len(results.algorithms) + 1))
        ax.set_xticklabels(results.algorithms, fontsize=6, color=TEXT_SECONDARY)
        if positive:
            ax.set_yscale("log")
        ax.set_title(function, fontsize=8, color=TEXT_PRIMARY, loc="left")
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.grid(False, axis="x")
    label = "final error f - f*" if results.metric == "error" else "final objective value"
    fig.supylabel(label, fontsize=8, color=TEXT_SECONDARY)
    fig.tight_layout(rect=(0.01, 0.0, 1.0, 1.0))
    return fig


def friedman_figure(results_dir: str | Path) -> Figure:
    """Friedman mean rank of every algorithm with the Nemenyi critical difference."""
    results = stats.load_results(results_dir)
    outcome = stats.friedman_test(results.matrix("mean"))
    n = len(results.algorithms)
    fig, ax = plt.subplots(figsize=(5.2, 0.45 * n + 1.3))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    positions = np.arange(n)[::-1]
    # Every bar is labelled on the axis; with more algorithms than categorical slots a
    # single colour is used instead of inventing further hues.
    colors = [_color(i) for i in range(n)] if n <= len(SERIES_COLORS) else SERIES_COLORS[0]
    ax.barh(
        positions,
        outcome.mean_ranks,
        height=0.5,
        color=colors,
        edgecolor=SURFACE,
        linewidth=1.5,
    )
    for y, rank in zip(positions, outcome.mean_ranks, strict=True):
        ax.text(rank + 0.04, y, f"{rank:.2f}", va="center", fontsize=8, color=TEXT_PRIMARY)
    ax.set_yticks(positions)
    ax.set_yticklabels(results.algorithms, fontsize=8, color=TEXT_PRIMARY)
    ax.set_xlim(0.0, n + 0.5)
    ax.grid(True, axis="x", color=GRID, linewidth=0.6)
    ax.grid(False, axis="y")
    ax.set_xlabel("Friedman mean rank (lower is better)", fontsize=8, color=TEXT_SECONDARY)
    title = f"{outcome.n_functions} functions, Friedman p = {outcome.p_value:.2E}"
    if 2 <= n <= 10:
        cd = stats.nemenyi_critical_difference(n, outcome.n_functions)
        best = float(outcome.mean_ranks.min())
        ax.plot([best, best + cd], [-0.75, -0.75], color=TEXT_PRIMARY, linewidth=1.5)
        ax.text(
            best + cd + 0.05,
            -0.75,
            f"CDV = {cd:.3f}",
            va="center",
            fontsize=7,
            color=TEXT_SECONDARY,
        )
        ax.set_ylim(-1.2, n - 0.4)
    ax.set_title(title, fontsize=8, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    return fig


def rank_sum_figure(results_dir: str | Path, reference: str | None = None) -> Figure:
    """Number of functions on which ``reference`` is better / similar / worse."""
    results = stats.load_results(results_dir)
    if reference is None:
        reference = "EDECO" if "EDECO" in results.algorithms else results.algorithms[-1]
    counts = stats.rank_sum_counts(results, reference)
    names = list(counts)
    fig, ax = plt.subplots(figsize=(1.5 * len(names) + 2.4, 3.0))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    width = 0.24
    for k, outcome in enumerate(("+", "=", "-")):
        x = np.arange(len(names)) + (k - 1) * (width + 0.02)
        heights = [counts[name][outcome] for name in names]
        ax.bar(
            x,
            heights,
            width=width,
            color=OUTCOME_COLORS[outcome],
            label=f"{outcome}  {OUTCOME_LABELS[outcome]}",
        )
        for xi, height in zip(x, heights, strict=True):
            ax.text(xi, height + 0.3, str(height), ha="center", fontsize=7, color=TEXT_PRIMARY)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels([f"vs. {name}" for name in names], fontsize=8, color=TEXT_PRIMARY)
    ax.set_ylim(0, len(results.functions) * 1.12)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.grid(False, axis="x")
    ax.set_ylabel("number of functions", fontsize=8, color=TEXT_SECONDARY)
    ax.set_title(
        f"Wilcoxon rank-sum test of {reference} (alpha = 0.05)",
        fontsize=8,
        color=TEXT_PRIMARY,
        loc="left",
    )
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3)
    fig.tight_layout()
    return fig


def write_all_figures(results_dir: str | Path, dpi: int = 150) -> list[Path]:
    """Render every figure into ``<results_dir>/figures`` and return the file paths."""
    results_dir = Path(results_dir)
    out_dir = results_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "friedman_ranks": friedman_figure(results_dir),
        "rank_sum": rank_sum_figure(results_dir),
    }
    if len(stats.load_results(results_dir).algorithms) <= len(SERIES_COLORS):
        figures["convergence"] = convergence_figure(results_dir)
        figures["boxplots"] = boxplot_figure(results_dir)
    else:
        logger.warning(
            "more than %d algorithms: convergence curves and box plots are skipped",
            len(SERIES_COLORS),
        )
    written = []
    for name, fig in figures.items():
        path = out_dir / f"{name}.png"
        fig.savefig(path, dpi=dpi, facecolor=fig.get_facecolor())
        plt.close(fig)
        written.append(path)
    return written
