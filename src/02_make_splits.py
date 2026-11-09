"""Build row groups and the frozen outer CV folds, with checks and a report.

Writes data/processed/groups.csv.gz, data/processed/outer_splits.csv.gz,
results/split_report.json and results/split_report.md. No model is trained.
"""
import gzip
import hashlib
import itertools
import json
import re

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import KFold

from common import (
    GROUPS_FILE,
    N_FOLDS,
    PROCESSED,
    RESULTS,
    SCHEMES,
    SEEDS,
    SPLITS_FILE,
    load_raw,
    seeded_group_folds,
    split_column,
)

REPORT_JSON = RESULTS / "split_report.json"
REPORT_MD = RESULTS / "split_report.md"
ROUND_DECIMALS = 6
MAX_FOLD_RATIO = 1.3
GROUPED_SCHEME_COLUMN = {"formula": "formula_group", "element": "element_group"}
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def row_ids(values):
    """Integer id per row, equal exactly when two rows of `values` are identical."""
    _, ids = np.unique(values, axis=0, return_inverse=True)
    return np.asarray(ids).ravel()


def row_ids_from_bytes(values):
    """Second, independent way to compute row_ids, used only to check the first."""
    keys = np.array([row.tobytes() for row in values + 0.0], dtype=object)
    return pd.factorize(keys)[0]


def linked_components(*key_columns):
    """Connected components of rows linked when any key column matches.

    Each key value becomes a node joined to its rows, so rows sharing a value in
    any column end up connected. Ids are numbered by first appearance.
    """
    n_rows = len(key_columns[0])
    row_nodes, key_nodes = [], []
    offset = n_rows
    for ids in key_columns:
        row_nodes.append(np.arange(n_rows))
        key_nodes.append(ids + offset)
        offset += int(ids.max()) + 1
    row_nodes = np.concatenate(row_nodes)
    key_nodes = np.concatenate(key_nodes)
    edges = np.ones(len(row_nodes), dtype=np.int8)
    graph = coo_matrix((edges, (row_nodes, key_nodes)), shape=(offset, offset))
    _, labels = connected_components(graph, directed=False)
    return pd.factorize(labels[:n_rows])[0]


def build_groups(X, composition):
    """Return the group table and the key arrays the checks and report reuse."""
    amounts = composition.to_numpy(dtype=float)
    fractions = amounts / amounts.sum(axis=1, keepdims=True)
    keys = {
        "raw_composition": row_ids(amounts),
        "normalized_composition": row_ids(np.round(fractions, ROUND_DECIMALS)),
        "exact_x": row_ids(X.to_numpy(dtype=float)),
        "element_set": row_ids(amounts > 0),
    }
    formula_group = linked_components(keys["normalized_composition"], keys["exact_x"])
    element_group = linked_components(keys["element_set"], formula_group)
    groups = pd.DataFrame(
        {
            "row_id": np.arange(len(X)),
            "formula_group": formula_group,
            "element_group": element_group,
        }
    )
    return groups, keys


def build_splits(groups):
    """Return one fold-id column per scheme and seed."""
    n_rows = len(groups)
    splits = {"row_id": np.arange(n_rows)}
    for scheme, seed in itertools.product(SCHEMES, SEEDS):
        if scheme == "random":
            folds = np.full(n_rows, -1, dtype=int)
            splitter = KFold(N_FOLDS, shuffle=True, random_state=seed)
            for fold, (_, test) in enumerate(splitter.split(np.zeros(n_rows))):
                assert (folds[test] == -1).all(), "row placed in two test folds"
                folds[test] = fold
        else:
            column = GROUPED_SCHEME_COLUMN[scheme]
            folds = seeded_group_folds(groups[column].to_numpy(), N_FOLDS, seed)
        splits[split_column(scheme, seed)] = folds
    return pd.DataFrame(splits)


def spans_one_value(owner, value):
    """True when every id in `owner` maps to a single id in `value`."""
    frame = pd.DataFrame({"owner": owner, "value": value})
    return bool((frame.groupby("owner")["value"].nunique() == 1).all())


def check_groups(groups, keys, X):
    """Assert the groups follow their definitions."""
    formula = groups["formula_group"].to_numpy()
    element = groups["element_group"].to_numpy()
    assert (groups["row_id"].to_numpy() == np.arange(len(groups))).all()
    assert spans_one_value(keys["normalized_composition"], formula)
    assert spans_one_value(keys["exact_x"], formula)
    assert spans_one_value(keys["element_set"], element)
    assert spans_one_value(formula, element), "formula groups must nest in element groups"
    independent = row_ids_from_bytes(X.to_numpy(dtype=float))
    assert independent.max() == keys["exact_x"].max()
    assert spans_one_value(independent, keys["exact_x"])


def check_splits(splits, groups, keys, X):
    """Assert the fold requirements for every scheme and seed."""
    exact_x = row_ids_from_bytes(X.to_numpy(dtype=float))
    n_rows = len(groups)
    assert (splits["row_id"].to_numpy() == np.arange(n_rows)).all()
    expected = [split_column(s, seed) for s, seed in itertools.product(SCHEMES, SEEDS)]
    assert list(splits.columns) == ["row_id"] + expected

    for scheme, seed in itertools.product(SCHEMES, SEEDS):
        column = split_column(scheme, seed)
        folds = splits[column].to_numpy()

        assert np.issubdtype(folds.dtype, np.integer), column
        assert set(np.unique(folds)) == set(range(N_FOLDS)), f"{column}: fold ids"
        times_tested = sum((folds == fold).astype(int) for fold in range(N_FOLDS))
        assert (times_tested == 1).all(), f"{column}: row not in exactly one test fold"
        assert np.bincount(folds, minlength=N_FOLDS).min() > 0, f"{column}: empty fold"

        if scheme == "random":
            continue
        group_column = GROUPED_SCHEME_COLUMN[scheme]
        assert spans_one_value(groups[group_column].to_numpy(), folds), f"{column}: group split"
        assert spans_one_value(keys["normalized_composition"], folds), f"{column}: composition"
        if scheme == "element":
            assert spans_one_value(keys["element_set"], folds), f"{column}: element set"
        for fold in range(N_FOLDS):
            shared = np.intersect1d(exact_x[folds == fold], exact_x[folds != fold])
            assert shared.size == 0, f"{column} fold {fold}: exact X row on both sides"


def write_gzip_csv(frame, path):
    """Write a CSV gzip with no timestamp, so identical data gives identical bytes."""
    path.write_bytes(gzip.compress(frame.to_csv(index=False).encode(), mtime=0))


def sha256_of(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def group_summary(group_ids):
    sizes = pd.Series(group_ids).value_counts().to_numpy()
    n_rows = len(group_ids)
    return {
        "n_groups": int(len(sizes)),
        "n_singletons": int((sizes == 1).sum()),
        "largest_group_size": int(sizes[0]),
        "share_rows_top1": round(float(sizes[:1].sum() / n_rows), 4),
        "share_rows_top5": round(float(sizes[:5].sum() / n_rows), 4),
        "share_rows_top10": round(float(sizes[:10].sum() / n_rows), 4),
    }


def top_element_sets(composition, y, n_top=10):
    names = composition.columns.to_numpy()
    labels = ["-".join(sorted(names[row])) for row in composition.to_numpy() > 0]
    table = (
        pd.DataFrame({"elements": labels, "tc": y.to_numpy()})
        .groupby("elements")["tc"]
        .agg(n_rows="size", mean_tc="mean")
        .sort_values(["n_rows", "mean_tc"], ascending=False)
        .head(n_top)
        .reset_index()
    )
    table["mean_tc"] = table["mean_tc"].round(2)
    return table.to_dict(orient="records"), len(set(labels))


def fold_report(splits, y):
    report = {}
    for scheme, seed in itertools.product(SCHEMES, SEEDS):
        column = split_column(scheme, seed)
        folds = splits[column].to_numpy()
        sizes = np.bincount(folds, minlength=N_FOLDS)
        mean_tc = y.groupby(folds).mean().to_numpy()
        report[column] = {
            "fold_sizes": [int(v) for v in sizes],
            "size_ratio_largest_to_smallest": round(float(sizes.max() / sizes.min()), 4),
            "min_fold_mean_tc": round(float(mean_tc.min()), 3),
            "max_fold_mean_tc": round(float(mean_tc.max()), 3),
        }
    return report


def independent_seed_ari(group_ids):
    """Adjusted Rand index expected if whole groups land in folds independently per seed.

    Rows of one group always stay together, so even unrelated seeds agree a little.
    """
    shares = pd.Series(group_ids).value_counts(normalize=True).to_numpy()
    same_group = float((shares**2).sum())
    same_fold = same_group + (1 - same_group) / N_FOLDS
    same_in_both = same_group + (1 - same_group) / N_FOLDS**2
    return (same_in_both - same_fold**2) / (same_fold - same_fold**2)


def seed_agreement(splits, groups):
    report = {}
    for scheme in ("formula", "element"):
        pairs = {}
        for a, b in itertools.combinations(SEEDS, 2):
            score = adjusted_rand_score(
                splits[split_column(scheme, a)], splits[split_column(scheme, b)]
            )
            pairs[f"s{a}_vs_s{b}"] = round(float(score), 5)
        values = list(pairs.values())
        report[scheme] = {
            "pairs": pairs,
            "min": min(values),
            "mean": round(float(np.mean(values)), 5),
            "max": max(values),
            "expected_if_seeds_independent": round(
                independent_seed_ari(groups[GROUPED_SCHEME_COLUMN[scheme]].to_numpy()), 5
            ),
        }
    return report


def random_scheme_leakage(splits, keys):
    """How many rows have an exact-input copy on the other side under `random`."""
    exact_x = keys["exact_x"]
    report = {}
    for seed in SEEDS:
        folds = splits[split_column("random", seed)].to_numpy()
        leaked = 0
        for fold in range(N_FOLDS):
            test = folds == fold
            leaked += int(np.isin(exact_x[test], exact_x[~test]).sum())
        report[f"s{seed}"] = {
            "test_rows_with_exact_x_copy_in_train": leaked,
            "share_of_rows": round(leaked / len(exact_x), 4),
        }
    return report


def count_reordered_strings(table):
    """Among strings sharing a raw composition, count reorderings of the first one."""
    reordered = 0
    for _, strings in table.drop_duplicates("string").groupby("raw")["string"]:
        tokens = [sorted(FORMULA_TOKEN.findall(text)) for text in strings]
        reordered += sum(other == tokens[0] for other in tokens[1:])
    return reordered


def formula_count_comparison(material, keys, groups):
    """Walk from unique formula strings to the formula-group count."""
    n_strings = int(material.nunique())
    n_raw = int(keys["raw_composition"].max() + 1)
    n_normalized = int(keys["normalized_composition"].max() + 1)
    n_groups = int(groups["formula_group"].max() + 1)
    n_exact_x = int(keys["exact_x"].max() + 1)
    table = pd.DataFrame(
        {
            "string": material.to_numpy(),
            "raw": keys["raw_composition"],
            "normalized": keys["normalized_composition"],
            "exact_x": keys["exact_x"],
            "formula_group": groups["formula_group"].to_numpy(),
        }
    )
    strings_per_group = table.groupby("formula_group")["string"].transform("nunique")
    n_reordered = count_reordered_strings(table)
    strings_per_raw = table.groupby("raw")["string"].nunique()
    raw_per_normalized = table.groupby("normalized")["raw"].nunique()
    x_per_normalized = table.groupby("normalized")["exact_x"].nunique()
    normalized_per_x = table.groupby("exact_x")["normalized"].nunique()
    return {
        "unique_formula_strings": n_strings,
        "unique_raw_composition_vectors": n_raw,
        "unique_normalized_compositions": n_normalized,
        "formula_groups": n_groups,
        "unique_exact_x_rows": n_exact_x,
        "strings_merged_same_composition_different_text": n_strings - n_raw,
        "strings_merged_element_order_only": n_reordered,
        "strings_merged_notation_variants": n_strings - n_raw - n_reordered,
        "raw_compositions_with_several_strings": int((strings_per_raw > 1).sum()),
        "compositions_merged_by_scale": n_raw - n_normalized,
        "normalized_compositions_with_several_scales": int((raw_per_normalized > 1).sum()),
        "groups_merged_by_exact_x_link": n_normalized - n_groups,
        "exact_x_rows_spanning_several_normalized_compositions": int((normalized_per_x > 1).sum()),
        "normalized_compositions_with_several_exact_x": int((x_per_normalized > 1).sum()),
        "formula_groups_with_several_strings": int(
            table.loc[strings_per_group > 1, "formula_group"].nunique()
        ),
        "rows_in_formula_groups_with_several_strings": int((strings_per_group > 1).sum()),
    }


def fold_flags(fold_stats):
    return [
        {"split": name, "ratio": stats["size_ratio_largest_to_smallest"]}
        for name, stats in fold_stats.items()
        if stats["size_ratio_largest_to_smallest"] > MAX_FOLD_RATIO
    ]


def write_markdown(report):
    grouping = report["grouping"]
    lines = [
        "# Split report",
        "",
        f"N = {report['n_rows']}, {report['n_folds']} folds, seeds {report['seeds']}.",
        "All checks in src/02_make_splits.py passed.",
        "",
        "## Groups",
        "",
        "| grouping | groups | singletons | largest | top 1 | top 5 | top 10 |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, row in grouping.items():
        lines.append(
            f"| {name} | {row['n_groups']} | {row['n_singletons']} "
            f"| {row['largest_group_size']} | {row['share_rows_top1']:.1%} "
            f"| {row['share_rows_top5']:.1%} | {row['share_rows_top10']:.1%} |"
        )

    comparison = report["formula_vs_strings"]
    lines += [
        "",
        "## Formula groups versus formula strings",
        "",
        f"{comparison['unique_formula_strings']} unique strings -> "
        f"{comparison['unique_raw_composition_vectors']} raw composition vectors "
        f"(-{comparison['strings_merged_same_composition_different_text']}: same amounts, "
        f"different text; {comparison['strings_merged_element_order_only']} element reorderings, "
        f"{comparison['strings_merged_notation_variants']} notation variants) -> "
        f"{comparison['unique_normalized_compositions']} normalized compositions "
        f"(-{comparison['compositions_merged_by_scale']}: scaled copies of one formula) -> "
        f"{comparison['formula_groups']} formula groups "
        f"(-{comparison['groups_merged_by_exact_x_link']}: exact-input link).",
        "",
        f"There are {comparison['unique_exact_x_rows']} unique input rows, slightly more "
        f"than the groups: {comparison['normalized_compositions_with_several_exact_x']} "
        "compositions have inputs that differ only by floating-point noise.",
        "",
        f"{comparison['formula_groups_with_several_strings']} formula groups "
        f"({comparison['rows_in_formula_groups_with_several_strings']} rows) contain more than "
        "one string, so folds grouped by string could split identical inputs.",
        "",
        "## Largest element sets",
        "",
        "| elements | rows | mean Tc (K) |",
        "|---|---|---|",
    ]
    for row in report["top_element_sets"]:
        lines.append(f"| {row['elements']} | {row['n_rows']} | {row['mean_tc']:.2f} |")

    lines += [
        "",
        f"Distinct element sets: {report['n_element_sets']}.",
        "",
        "## Folds",
        "",
        "| split | fold sizes | largest/smallest | min fold mean Tc | max fold mean Tc |",
        "|---|---|---|---|---|",
    ]
    for name, row in report["folds"].items():
        sizes = ", ".join(str(v) for v in row["fold_sizes"])
        lines.append(
            f"| {name} | {sizes} | {row['size_ratio_largest_to_smallest']:.3f} "
            f"| {row['min_fold_mean_tc']:.2f} | {row['max_fold_mean_tc']:.2f} |"
        )

    lines += ["", "## Agreement between seeds (adjusted Rand index)", ""]
    lines += ["| scheme | min | mean | max | expected if independent |", "|---|---|---|---|---|"]
    for scheme, row in report["seed_ari"].items():
        lines.append(
            f"| {scheme} | {row['min']:.4f} | {row['mean']:.4f} | {row['max']:.4f} "
            f"| {row['expected_if_seeds_independent']:.4f} |"
        )
    lines += ["", "Pairwise values for every seed pair are in split_report.json."]

    leak = report["random_scheme_leakage"]
    shares = [row["share_of_rows"] for row in leak.values()]
    lines += [
        "",
        "## Random folds",
        "",
        f"Under `random`, {min(shares):.1%} to {max(shares):.1%} of rows have an exact-input "
        "copy on the other side of their fold.",
        "",
        "## Flags",
        "",
    ]
    if report["flags"]:
        lines += [f"- {row['split']}: largest/smallest fold = {row['ratio']:.3f}" for row in report["flags"]]
    else:
        lines.append(f"No split has largest/smallest fold above {MAX_FOLD_RATIO}.")

    lines += ["", "## SHA-256", ""]
    lines += [f"- `{name}`: `{digest}`" for name, digest in report["sha256"].items()]
    REPORT_MD.write_text("\n".join(lines) + "\n")


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    X, y, composition, material = load_raw()

    groups, keys = build_groups(X, composition)
    check_groups(groups, keys, X)

    splits = build_splits(groups)
    check_splits(splits, groups, keys, X)
    regrouped, _ = build_groups(X, composition)
    assert regrouped.equals(groups), "group assignment is not deterministic"
    assert build_splits(regrouped).equals(splits), "fold assignment is not deterministic"

    write_gzip_csv(groups, GROUPS_FILE)
    write_gzip_csv(splits, SPLITS_FILE)
    assert pd.read_csv(GROUPS_FILE).equals(groups), "groups file does not read back"
    assert pd.read_csv(SPLITS_FILE).equals(splits), "splits file does not read back"

    top_sets, n_element_sets = top_element_sets(composition, y)
    fold_stats = fold_report(splits, y)
    report = {
        "n_rows": int(len(X)),
        "n_folds": N_FOLDS,
        "seeds": SEEDS,
        "schemes": SCHEMES,
        "grouping": {
            "formula_group": group_summary(groups["formula_group"].to_numpy()),
            "element_group": group_summary(groups["element_group"].to_numpy()),
        },
        "n_element_sets": n_element_sets,
        "top_element_sets": top_sets,
        "folds": fold_stats,
        "seed_ari": seed_agreement(splits, groups),
        "random_scheme_leakage": random_scheme_leakage(splits, keys),
        "formula_vs_strings": formula_count_comparison(material, keys, groups),
        "flag_threshold_largest_to_smallest": MAX_FOLD_RATIO,
        "flags": fold_flags(fold_stats),
        "sha256": {
            GROUPS_FILE.name: sha256_of(GROUPS_FILE),
            SPLITS_FILE.name: sha256_of(SPLITS_FILE),
        },
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report)
    print(f"Wrote {GROUPS_FILE}, {SPLITS_FILE}, {REPORT_JSON}, {REPORT_MD}")


if __name__ == "__main__":
    main()
