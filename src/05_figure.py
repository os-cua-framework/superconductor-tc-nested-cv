"""Plot CV RMSE by model and outer-CV scheme from results/summary_by_model.csv.

Run src/04_summarize.py first. Models and marks without complete seeds are left
out, so the script works on partial results and can be rerun at any time.
"""
import math
import os
import shutil
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

from common import FIGURES, N_FOLDS, RESULTS, ROOT, SCHEMES

SUMMARY = RESULTS / "summary_by_model.csv"
FIGURE = FIGURES / "baseline.png"
SUBMISSION = ROOT / "submission" / "baseline.png"

MODEL_LABELS = {
    "mean": "Mean",
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "rf": "Random forest",
}
# Colours are the validated first three categorical slots; shape carries identity
# in greyscale and for readers who cannot separate the hues.
SCHEME_STYLES = {
    "random": {"label": "Random folds", "color": "#2a78d6", "marker": "o", "size": 6.5},
    "formula": {"label": "Formula-grouped folds", "color": "#eb6834", "marker": "s", "size": 6},
    "element": {"label": "Element-set folds", "color": "#1baf7a", "marker": "^", "size": 7},
}
OFFSETS = {"random": -0.24, "formula": 0.0, "element": 0.24}

PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def seed_note(table, models):
    """Return a sentence saying how many complete seeds the marks use."""
    most = int(table["n_seeds"].max())
    if table["n_seeds"].nunique() == 1:
        note = f"All marks use {most} seed{'s' if most != 1 else ''} ({most * N_FOLDS} folds)."
    else:
        fewer = table[table["n_seeds"] < most]
        listing = "; ".join(
            f"{MODEL_LABELS[r.model]} {r.scheme} {r.n_seeds}" for r in fewer.itertuples()
        )
        note = f"Marks use {most} seeds except: {listing}."
    if len(table) < len(models) * len(SCHEMES):
        note += " A missing mark has no complete seed yet."
    return note


def draw(table, models, path):
    """Draw one figure and write it to path as PNG."""
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"]})
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    fig.subplots_adjust(left=0.095, right=0.985, top=0.88, bottom=0.225)

    positions = {model: i for i, model in enumerate(models)}
    for scheme in SCHEMES:
        style = SCHEME_STYLES[scheme]
        part = table[table["scheme"] == scheme]
        ax.errorbar(
            [positions[m] + OFFSETS[scheme] for m in part["model"]],
            part["rmse_mean"],
            yerr=part["rmse_sd"],
            label=style["label"],
            fmt=style["marker"],
            markersize=style["size"],
            color=style["color"],
            markeredgecolor="white",
            markeredgewidth=0.7,
            elinewidth=1.3,
            capsize=5,
            capthick=1.3,
            zorder=3,
        )

    top = table["rmse_mean"] + table["rmse_sd"]
    ax.set_ylim(0, math.ceil((top.max() + 1) / 5) * 5)
    ax.set_xlim(-0.55, len(models) - 0.45)
    ax.yaxis.set_major_locator(MultipleLocator(10))
    ax.set_xticks(range(len(models)), [MODEL_LABELS[m] for m in models])
    ax.set_ylabel("CV RMSE (K)", fontsize=11.5, color=PRIMARY)
    ax.tick_params(axis="x", length=0, labelsize=11.5, labelcolor=PRIMARY, pad=7)
    ax.tick_params(axis="y", length=0, labelsize=10.5, labelcolor=SECONDARY)
    ax.yaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)

    ax.legend(
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        borderpad=0.0,
        ncol=3,
        frameon=False,
        fontsize=10.5,
        handletextpad=0.3,
        columnspacing=1.6,
        labelcolor=PRIMARY,
    )

    caption = (
        f"Markers show mean test RMSE over the {N_FOLDS} outer folds of a seed, averaged over "
        f"seeds. Error bars show ±1 SD across the {N_FOLDS} folds within a seed, averaged "
        f"over seeds; they are not confidence intervals. {seed_note(table, models)}"
    )
    fig.text(0.02, 0.135, textwrap.fill(caption, 118), fontsize=8, color=SECONDARY, va="top")

    fig.savefig(path, dpi=200, format="png", facecolor="white")
    plt.close(fig)


def main():
    if not SUMMARY.exists():
        print(f"{SUMMARY} not found; run src/04_summarize.py first")
        return
    summary = pd.read_csv(SUMMARY)
    table = summary[summary["n_seeds"] > 0]
    models = [m for m in MODEL_LABELS if m in set(table["model"])]
    if not models:
        print("no model has a complete seed yet; nothing to plot")
        return

    FIGURES.mkdir(parents=True, exist_ok=True)
    SUBMISSION.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIGURE.with_name(FIGURE.name + ".tmp")
    draw(table[table["model"].isin(models)], models, tmp)
    os.replace(tmp, FIGURE)
    tmp = SUBMISSION.with_name(SUBMISSION.name + ".tmp")
    shutil.copyfile(FIGURE, tmp)
    os.replace(tmp, SUBMISSION)
    print(f"wrote {FIGURE} and {SUBMISSION} with models: {', '.join(models)}")


if __name__ == "__main__":
    main()
