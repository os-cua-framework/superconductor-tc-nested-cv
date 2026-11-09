"""Summarize the finished nested-CV jobs into tables and a short report.

Reads whatever results/jobs/*.json exist, so it works on partial results and can
be rerun at any time. Every output is replaced through a temporary file, so a
reader never sees a half-written table.
"""
import json
import math
import os

import numpy as np
import pandas as pd

from common import JOBS, N_FOLDS, RESULTS, SCHEMES, SEEDS

MODELS = ["mean", "ols", "ridge", "lasso", "rf"]
LINEAR = ["ols", "ridge", "lasso"]
TUNED = ["ridge", "lasso", "rf"]
PENALTY = {"ridge": "ridge__alpha", "lasso": "lasso__alpha"}
REQUIRED = {
    "model", "scheme", "seed", "fold", "rmse", "r2", "best_index", "n_settings",
    "best_setting", "at_grid_edge", "convergence_warnings",
}
SELECTED_COLUMNS = [
    "row_type", "model", "scheme", "setting", "penalty", "count", "n_folds",
    "penalty_min", "penalty_median", "penalty_max", "folds_at_grid_edge",
    "convergence_warnings",
]

FOLD_METRICS = RESULTS / "fold_metrics.csv"
SUMMARY = RESULTS / "summary_by_model.csv"
PAIRED_GAP = RESULTS / "paired_gap.csv"
SCHEME_CHANGE = RESULTS / "scheme_change.csv"
SELECTED = RESULTS / "selected_params.csv"
REPORT = RESULTS / "summary.md"


def study_order(column):
    """Sort key that puts models and schemes in the study's order."""
    if column.name == "model":
        return column.map(MODELS.index)
    if column.name == "scheme":
        return column.map(SCHEMES.index)
    return column


def write_text(path, text):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def write_csv(frame, path, **options):
    tmp = path.with_name(path.name + ".tmp")
    frame.to_csv(tmp, index=False, **options)
    os.replace(tmp, path)


def is_finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def load_jobs():
    """Return all job records in study order and the problems found while reading."""
    records, problems = [], []
    for path in sorted(JOBS.glob("*.json")):
        try:
            record = json.loads(path.read_text())
            missing = REQUIRED - record.keys()
        except (OSError, ValueError, AttributeError) as error:
            problems.append(f"{path.name}: unreadable, skipped ({error})")
            continue
        if missing:
            problems.append(f"{path.name}: missing {sorted(missing)}, skipped")
            continue
        model, scheme, seed, fold = (record[k] for k in ("model", "scheme", "seed", "fold"))
        if path.stem != f"{model}__{scheme}__s{seed}__f{fold}":
            problems.append(f"{path.name}: fields say {model}/{scheme}/s{seed}/f{fold}, skipped")
            continue
        if model not in MODELS or scheme not in SCHEMES or seed not in SEEDS or fold not in range(N_FOLDS):
            problems.append(f"{path.name}: outside the study design, skipped")
            continue
        if not (is_finite(record["rmse"]) and is_finite(record["r2"])):
            problems.append(f"{path.name}: rmse or r2 is not finite, treated as unfinished")
        records.append(record)
    if not records:
        return pd.DataFrame(), problems
    jobs = pd.DataFrame(records).sort_values(["model", "scheme", "seed", "fold"], key=study_order)
    jobs = jobs.reset_index(drop=True)
    if "rf_trees" in jobs:
        trees = sorted(jobs.loc[jobs["model"] == "rf", "rf_trees"].dropna().unique())
        if len(trees) > 1:
            problems.append(f"rf jobs mix different forest sizes: {trees}")
    return jobs, problems


def fold_metrics(jobs):
    """Return the job records with nested fields stored as JSON strings."""
    table = jobs.copy()
    for column in ("best_setting", "inner_rmse"):
        if column in table:
            table[column] = table[column].map(lambda v: json.dumps(v, sort_keys=True))
    return table


def seed_stats(jobs):
    """Return per-seed statistics for every model, scheme and seed, with its fold count."""
    return (
        jobs.groupby(["model", "scheme", "seed"])
        .agg(
            folds=("fold", "nunique"),
            mean=("rmse", "mean"),
            sd=("rmse", lambda s: s.std(ddof=1)),
            r2=("r2", "mean"),
        )
        .reset_index()
    )


def summary_by_model(jobs):
    """Return one row per model and scheme, using only seeds with all folds finished."""
    seeds = seed_stats(jobs)
    complete = seeds[seeds["folds"] == N_FOLDS]
    rows = []
    for (model, scheme), _ in jobs.groupby(["model", "scheme"]):
        used = complete[(complete["model"] == model) & (complete["scheme"] == scheme)]
        rows.append({
            "model": model,
            "scheme": scheme,
            "n_seeds": len(used),
            "n_folds": len(used) * N_FOLDS,
            "rmse_mean": used["mean"].mean(),
            "rmse_sd": used["sd"].mean(),
            "rmse_se": (used["sd"] / math.sqrt(N_FOLDS)).mean(),
            "seed_sd": used["mean"].std(ddof=1),
            "r2_mean": used["r2"].mean(),
        })
    table = pd.DataFrame(rows).sort_values(["model", "scheme"], key=study_order)
    return table.reset_index(drop=True)


def paired_gap(jobs, summary):
    """Return per-scheme fold-paired gaps (best linear minus rf), or None without rf."""
    if not (jobs["model"] == "rf").any():
        return None
    rows = []
    for scheme in SCHEMES:
        candidates = summary[
            (summary["scheme"] == scheme)
            & summary["model"].isin(LINEAR)
            & (summary["n_seeds"] > 0)
        ]
        best = candidates.loc[candidates["rmse_mean"].idxmin(), "model"] if len(candidates) else ""
        in_scheme = jobs[jobs["scheme"] == scheme]
        pairs = in_scheme.loc[in_scheme["model"] == best, ["seed", "fold", "rmse"]].merge(
            in_scheme.loc[in_scheme["model"] == "rf", ["seed", "fold", "rmse"]],
            on=["seed", "fold"],
            suffixes=("_linear", "_rf"),
        )
        gap = pairs["rmse_linear"] - pairs["rmse_rf"]
        by_seed = gap.groupby(pairs["seed"])
        complete_sds = by_seed.std(ddof=1)[by_seed.size() == N_FOLDS]
        gap_mean = gap.mean()
        gap_sd = gap.std(ddof=1)
        rows.append({
            "scheme": scheme,
            "best_linear": best,
            "n_folds": len(pairs),
            "seeds_used": ";".join(str(s) for s in sorted(pairs["seed"].unique())),
            "n_complete_seeds": len(complete_sds),
            "gap_mean": gap_mean,
            "gap_sd": gap_sd,
            "gap_se": (complete_sds / math.sqrt(N_FOLDS)).mean(),
            "tie": None if np.isnan(gap_sd) else bool(abs(gap_mean) <= gap_sd),
        })
    table = pd.DataFrame(rows)
    table["tie"] = table["tie"].astype(object)
    return table


def scheme_change(summary):
    """Return rmse_mean per scheme and the change from random folds, one row per model."""
    rows = []
    for model in summary["model"].unique():
        part = summary[summary["model"] == model].set_index("scheme")
        row = {"model": model}
        for scheme in SCHEMES:
            row[f"rmse_mean_{scheme}"] = part["rmse_mean"].get(scheme, np.nan)
        row["change_random_to_formula"] = row["rmse_mean_formula"] - row["rmse_mean_random"]
        row["change_random_to_element"] = row["rmse_mean_element"] - row["rmse_mean_random"]
        for scheme in SCHEMES:
            row[f"n_seeds_{scheme}"] = part["n_seeds"].get(scheme, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def selected_params(jobs):
    """Return settings chosen, per-scheme penalty and edge counts, and per-model warnings.

    row_type "setting" holds one chosen setting and its count, "scheme" holds the
    per-scheme fold count, penalty range and grid-edge count, and "model" holds the
    total convergence warnings. Other columns are blank on each row type.
    """
    rows = []
    for model in TUNED:
        for scheme in SCHEMES:
            part = jobs[(jobs["model"] == model) & (jobs["scheme"] == scheme)]
            if part.empty:
                continue
            row = {
                "row_type": "scheme",
                "model": model,
                "scheme": scheme,
                "n_folds": len(part),
                "folds_at_grid_edge": int(part["at_grid_edge"].sum()),
            }
            if model in PENALTY:
                penalties = part["best_setting"].map(lambda s: s[PENALTY[model]])
                row.update(
                    penalty_min=penalties.min(),
                    penalty_median=penalties.median(),
                    penalty_max=penalties.max(),
                )
            rows.append(row)
            labels = part["best_setting"].map(lambda s: json.dumps(s, sort_keys=True))
            counts = labels.value_counts().sort_index().sort_values(ascending=False, kind="stable")
            for label, count in counts.items():
                setting = json.loads(label)
                rows.append({
                    "row_type": "setting",
                    "model": model,
                    "scheme": scheme,
                    "setting": label,
                    "penalty": setting.get(PENALTY.get(model), np.nan),
                    "count": int(count),
                })
    for model in jobs["model"].unique():
        part = jobs[jobs["model"] == model]
        rows.append({
            "row_type": "model",
            "model": model,
            "scheme": "all",
            "n_folds": len(part),
            "convergence_warnings": int(part["convergence_warnings"].sum()),
        })
    return pd.DataFrame(rows).reindex(columns=SELECTED_COLUMNS)


def format_cell(value, digits):
    if value is None or value is pd.NA or (isinstance(value, float) and math.isnan(value)):
        return "-"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{digits}f}"
    return str(value)


def markdown_table(frame, digits=3):
    """Return a Markdown table; missing values print as a dash."""
    lines = [
        "| " + " | ".join(frame.columns) + " |",
        "|" + "---|" * len(frame.columns),
    ]
    for row in frame.itertuples(index=False):
        lines.append("| " + " | ".join(format_cell(v, digits) for v in row) + " |")
    return "\n".join(lines)


def short_setting(label):
    """Return a compact readable form of a setting stored as JSON."""
    parts = []
    for key, value in json.loads(label).items():
        name = key.split("__")[-1]
        parts.append(f"{name}={value:.3g}" if isinstance(value, float) else f"{name}={value}")
    return ", ".join(parts) if parts else "none"


def selection_table(jobs, selected):
    """Return the per-scheme selection summary shown in the report."""
    rows = []
    settings = selected[selected["row_type"] == "setting"]
    for item in selected[selected["row_type"] == "scheme"].itertuples():
        part = jobs[(jobs["model"] == item.model) & (jobs["scheme"] == item.scheme)]
        top = settings[(settings["model"] == item.model) & (settings["scheme"] == item.scheme)].iloc[0]
        rows.append({
            "model": item.model,
            "scheme": item.scheme,
            "folds": int(item.n_folds),
            "penalty min": "-" if np.isnan(item.penalty_min) else f"{item.penalty_min:.3g}",
            "median": "-" if np.isnan(item.penalty_median) else f"{item.penalty_median:.3g}",
            "max": "-" if np.isnan(item.penalty_max) else f"{item.penalty_max:.3g}",
            "at grid edge": int(item.folds_at_grid_edge),
            "at low edge": int((part["best_index"] == 0).sum()),
            "at high edge": int((part["best_index"] == part["n_settings"] - 1).sum()),
            "most common setting (count)": f"{short_setting(top['setting'])} ({int(top['count'])})",
        })
    return pd.DataFrame(rows)


def incomplete_seeds(jobs):
    """Return (model, scheme, seed, folds finished) for seeds left out of the summary."""
    seeds = seed_stats(jobs)
    partial = seeds[seeds["folds"] < N_FOLDS].sort_values(["model", "scheme", "seed"], key=study_order)
    return [(r.model, r.scheme, r.seed, r.folds) for r in partial.itertuples()]


def report(jobs, usable, summary, gaps, changes, selected, problems):
    """Return the Markdown report text."""
    expected = len(SCHEMES) * len(SEEDS) * N_FOLDS
    counts = usable["model"].value_counts()
    finished = ", ".join(f"{m} {int(counts.get(m, 0))}" for m in MODELS)
    lines = [
        "# Nested CV summary",
        "",
        f"{len(jobs)} job records read. Finished jobs per model (of {expected}): {finished}.",
        "",
        "## Test RMSE by model and scheme (K)",
        "",
        f"Only seeds with all {N_FOLDS} folds finished are used. rmse_sd and rmse_se are computed",
        "within each seed and then averaged over seeds; seed_sd is the SD of the seed means.",
        "",
        markdown_table(summary),
        "",
        "## Paired gap: best linear model minus random forest",
        "",
    ]
    if gaps is None:
        lines.append("No random forest jobs have finished yet.")
    else:
        lines += [
            "A positive gap means the forest has the lower RMSE. Pairs are the same (seed, fold).",
            "tie is true when the absolute mean gap is within one SD of the fold gaps. best_linear",
            "is picked on the same outer-test RMSE used for the gap, which slightly favours it.",
            "",
            markdown_table(gaps),
        ]
    lines += [
        "",
        "## Change in RMSE when folds are grouped (K)",
        "",
        "Positive means RMSE rises compared with random folds.",
        "",
        markdown_table(changes),
        "",
        "## Selected settings",
        "",
    ]
    if (selected["row_type"] == "scheme").any():
        lines.append(markdown_table(selection_table(usable, selected)))
    else:
        lines.append("No tuned model has finished yet.")
    totals = selected[selected["row_type"] == "model"]
    warnings = ", ".join(f"{r.model} {int(r.convergence_warnings)}" for r in totals.itertuples())
    lines += ["", f"Convergence warnings, total per model: {warnings}.", "", "## Notes", ""]

    notes = []
    for model in PENALTY:
        part = usable[usable["model"] == model]
        low = int((part["best_index"] == 0).sum())
        high = int((part["best_index"] == part["n_settings"] - 1).sum())
        if low or high:
            notes.append(
                f"{model} chose the smallest penalty in {low} of {len(part)} folds and the "
                f"largest in {high}; the best penalty for those folds may lie outside the grid."
            )
    if (usable["model"] == "rf").any():
        notes.append(
            "For rf, at_grid_edge only says the chosen setting was the first or last of the "
            "random draws. Those draws are unordered, so it does not mean a search boundary."
        )
    partial = incomplete_seeds(usable)
    if partial:
        listing = "; ".join(f"{m}/{sc} s{s} ({n}/{N_FOLDS})" for m, sc, s, n in partial)
        notes.append(f"Seeds left out as incomplete: {listing}.")
    if len(summary["n_seeds"].unique()) > 1:
        notes.append("Rows use different numbers of seeds, so comparisons between them are provisional.")
    notes.extend(f"Record problem: {p}" for p in problems)
    if not problems:
        notes.append("No problems found while reading the job records.")
    lines += [f"- {note}" for note in notes]
    return "\n".join(lines) + "\n"


def main():
    jobs, problems = load_jobs()
    if jobs.empty:
        print("no finished jobs found in", JOBS)
        for problem in problems:
            print(problem)
        return

    usable = jobs[np.isfinite(jobs["rmse"]) & np.isfinite(jobs["r2"])]
    summary = summary_by_model(usable)
    gaps = paired_gap(usable, summary)
    changes = scheme_change(summary)
    selected = selected_params(usable)

    write_csv(fold_metrics(jobs), FOLD_METRICS)
    write_csv(summary, SUMMARY, float_format="%.6g")
    if gaps is None:
        PAIRED_GAP.unlink(missing_ok=True)
    else:
        write_csv(gaps, PAIRED_GAP, float_format="%.6g")
    write_csv(changes, SCHEME_CHANGE, float_format="%.6g")
    write_csv(selected, SELECTED, float_format="%.6g")
    write_text(REPORT, report(jobs, usable, summary, gaps, changes, selected, problems))

    print(f"read {len(jobs)} jobs, {len(problems)} problems")
    print(summary.to_string(index=False))
    if gaps is not None:
        print(gaps.to_string(index=False))


if __name__ == "__main__":
    main()
