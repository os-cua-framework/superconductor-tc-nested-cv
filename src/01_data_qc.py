"""Data quality report for the superconductor data (UCI 464, Hamidieh 2018).

Reads the two raw files, changes nothing, drops no rows, and writes
results/data_qc.json and results/data_qc.md. Row numbers in the output are
0-based positions in the raw files.
"""
import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from common import RAW, RESULTS, TARGET, load_raw

MAX_EXAMPLES = 10
TC_TOLERANCE = 1e-9
COMPOSITION_DECIMALS = 6
LARGE_TC_RANGE_K = 10.0
# X differences below this many column SDs are treated as float noise.
X_NOISE_SD = 1e-8
X_NOISE_DECIMALS = 8
# Coarse rounding of X in SD units, used only to look for near-identical inputs.
NEAR_MISS_DECIMALS = 3

PROPOSAL = {
    "rows": 21263,
    "inputs": 81,
    "element_columns": 86,
    "missing_values": 0,
    "tc_mean": 34.4,
    "tc_sd": 34.3,
    "pcs_for_90pct": 12,
    "unique_material_strings": 15542,
    "pct_rows_with_repeated_string": 37,
}


def group_ids(matrix):
    """Return an integer id per row such that equal rows share an id."""
    _, inverse = np.unique(matrix, axis=0, return_inverse=True)
    return inverse.ravel()


def round6(value):
    return round(float(value), 6)


def composition_dict(fractions_row, elements):
    return {e: round6(f) for e, f in zip(elements, fractions_row) if f > 0}


def pooled_rmse(tc, groups):
    """Return sqrt(SSW / (N - K)) and its parts, for K groups over N rows."""
    within = tc - tc.groupby(groups).transform("mean")
    ssw = float((within**2).sum())
    n_groups = int(pd.Series(groups).nunique())
    dof = int(len(tc) - n_groups)
    return {"rmse": float(np.sqrt(ssw / dof)), "ssw": ssw, "dof": dof, "n_groups": n_groups}


def member_rows(rows, column, value):
    return rows.loc[rows[column].to_numpy() == value]


def check_shape(X, y, composition, material):
    numeric_x = X.select_dtypes("number").to_numpy(dtype=float)
    numeric_c = composition.select_dtypes("number").to_numpy(dtype=float)
    return {
        "rows": len(X),
        "rows_unique_m": len(composition),
        "inputs": X.shape[1],
        "element_columns": composition.shape[1],
        "missing_values": {
            "X": int(X.isna().sum().sum()),
            "tc": int(y.isna().sum()),
            "element_columns": int(composition.isna().sum().sum()),
            "material": int(material.isna().sum()),
        },
        "infinite_values": {
            "X": int(np.isinf(numeric_x).sum()),
            "tc": int(np.isinf(y.to_numpy(dtype=float)).sum()),
            "element_columns": int(np.isinf(numeric_c).sum()),
        },
        "non_numeric_columns": {
            "X": [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])],
            "element_columns": [
                c for c in composition.columns if not pd.api.types.is_numeric_dtype(composition[c])
            ],
        },
        "constant_columns": {
            "X": [c for c in X.columns if X[c].nunique() <= 1],
            "element_columns_never_present": [
                c for c in composition.columns if composition[c].nunique() <= 1
            ],
        },
        "negative_element_entries": int((numeric_c < 0).sum()),
        "rows_with_empty_composition": int((numeric_c.sum(axis=1) == 0).sum()),
    }


def check_tc(y, material):
    top = y.sort_values(ascending=False, kind="stable").head(MAX_EXAMPLES)
    return {
        "mean": float(y.mean()),
        "sd_ddof1": float(y.std(ddof=1)),
        "min": float(y.min()),
        "median": float(y.median()),
        "max": float(y.max()),
        "n_tc_le_0": int((y <= 0).sum()),
        "top_10": [
            {"row": int(i), "material": material.iloc[i], "tc": float(y.iloc[i])} for i in top.index
        ],
    }


def check_alignment(X, y, composition, material, unique_m_tc):
    tc_differs = np.abs(unique_m_tc.to_numpy() - y.to_numpy()) > TC_TOLERANCE
    n_nonzero = (composition.to_numpy() != 0).sum(axis=1)
    n_elements = X["number_of_elements"].to_numpy()
    mismatch = np.flatnonzero(n_nonzero != n_elements)
    return {
        "n_rows_tc_differs": int(tc_differs.sum()),
        "n_rows_number_of_elements_differs": int(len(mismatch)),
        "number_of_elements_examples": [
            {
                "row": int(i),
                "material": material.iloc[i],
                "number_of_elements": int(n_elements[i]),
                "nonzero_element_columns": int(n_nonzero[i]),
            }
            for i in mismatch[:MAX_EXAMPLES]
        ],
    }


def check_pca(X):
    standardized = StandardScaler().fit_transform(X.to_numpy())
    singular_values = np.linalg.svd(standardized, compute_uv=False)
    cumulative = np.cumsum(singular_values**2) / np.sum(singular_values**2)

    def needed(share):
        return int(np.searchsorted(cumulative, share) + 1)

    k = needed(0.90)
    return {
        "pcs_for_90pct": k,
        "cumulative_variance_at_k_minus_1": float(cumulative[k - 2]),
        "cumulative_variance_at_k": float(cumulative[k - 1]),
        "pcs_for_95pct": needed(0.95),
        "pcs_for_99pct": needed(0.99),
    }


def check_formula_strings(composition, material):
    """Parse each formula, compare it to the element columns, and find bare 'O' tokens."""
    token = re.compile(r"([A-Z][a-z]?)([0-9.]*)")
    columns = {element: j for j, element in enumerate(composition.columns)}
    table = composition.to_numpy(dtype=float)
    n_disagree = 0
    bare_oxygen = []
    for i, formula in enumerate(material):
        tokens = token.findall(formula)
        if ("O", "") in tokens:
            bare_oxygen.append(i)
        amounts = defaultdict(float)
        for element, count in tokens:
            amounts[element] += float(count) if count else 1.0
        parsed = np.zeros(table.shape[1])
        known = all(element in columns for element in amounts)
        if known:
            for element, amount in amounts.items():
                parsed[columns[element]] = amount
        if not known or not np.allclose(parsed, table[i], rtol=1e-6, atol=1e-6):
            n_disagree += 1

    bare_oxygen = np.array(bare_oxygen, dtype=int)
    has_cu = composition["Cu"].to_numpy()[bare_oxygen] > 0
    common_strings = material.iloc[bare_oxygen].value_counts().head(MAX_EXAMPLES)
    return {
        "n_rows_string_disagrees_with_element_columns": n_disagree,
        "n_rows_bare_oxygen": int(len(bare_oxygen)),
        "pct_rows_bare_oxygen": float(100 * len(bare_oxygen) / len(material)),
        "n_rows_bare_oxygen_with_cu": int(has_cu.sum()),
        "n_rows_bare_oxygen_with_o_equal_1": int(
            (composition["O"].to_numpy()[bare_oxygen] == 1).sum()
        ),
        "most_common_bare_oxygen_strings": {k: int(v) for k, v in common_strings.items()},
        "note": (
            "A bare 'O' token is stored as O = 1 in the element columns. Formulas like Y1Ba2Cu3O "
            "are physically implausible at O = 1, so this most likely means unspecified oxygen."
        ),
    }


def check_repeats(X, composition, fractions, rows):
    elements = list(composition.columns)
    string_repeat = rows.groupby("material")["tc"].transform("size") > 1
    per_string = rows.groupby("material").agg(n_x=("x", "nunique"), n_comp=("composition", "nunique"))

    per_composition = rows.groupby("composition").agg(
        n_rows=("tc", "size"),
        n_strings=("material", "nunique"),
        n_x=("x", "nunique"),
        first_row=("row", "min"),
    )
    multi_string = per_composition[per_composition["n_strings"] > 1].sort_values(
        ["n_strings", "n_rows", "first_row"], ascending=[False, False, True]
    )

    string_examples = []
    for composition_id, info in multi_string.head(MAX_EXAMPLES).iterrows():
        members = member_rows(rows, "composition", composition_id)
        string_examples.append(
            {
                "composition": composition_dict(fractions[int(info["first_row"])], elements),
                "n_rows": int(info["n_rows"]),
                "strings": sorted(members["material"].unique()),
                "tc_min": round6(members["tc"].min()),
                "tc_max": round6(members["tc"].max()),
            }
        )

    # Spread of X within a composition, in SD units, avoids any rounding boundary.
    z = pd.DataFrame(X.to_numpy() / X.std(ddof=0).to_numpy())
    by_composition = z.groupby(rows["composition"].to_numpy())
    spread_sd = (by_composition.max() - by_composition.min()).max(axis=1)

    multi_x = per_composition[per_composition["n_x"] > 1].sort_values("first_row")
    x_examples = []
    for composition_id, info in multi_x.head(MAX_EXAMPLES).iterrows():
        members = member_rows(rows, "composition", composition_id)
        x_examples.append(
            {
                "rows": members["row"].astype(int).tolist(),
                "strings": sorted(members["material"].unique()),
                "n_distinct_x": int(info["n_x"]),
                "max_column_difference_in_sd": float(spread_sd.loc[composition_id]),
            }
        )

    x_compositions = rows.groupby("x")["composition"].nunique()
    multi_composition = x_compositions[x_compositions > 1]
    composition_examples = []
    for x_id in multi_composition.index[:MAX_EXAMPLES]:
        members = member_rows(rows, "x", x_id)
        composition_examples.append(
            {"rows": members["row"].astype(int).tolist(), "strings": sorted(members["material"].unique())}
        )

    rounded_x = np.round(X.to_numpy() / X.std(ddof=0).to_numpy(), NEAR_MISS_DECIMALS)
    near = rows.assign(near_x=group_ids(rounded_x))
    near_compositions = near.groupby("near_x")["composition"].nunique()
    near_examples = []
    for near_id in near_compositions[near_compositions > 1].index[:MAX_EXAMPLES]:
        members = near.loc[near["near_x"].to_numpy() == near_id]
        near_examples.append(
            {
                "rows": members["row"].astype(int).tolist(),
                "strings": members["material"].tolist(),
                "tc": [round6(t) for t in members["tc"]],
            }
        )

    largest = per_composition.sort_values(["n_rows", "first_row"], ascending=[False, True]).head(MAX_EXAMPLES)
    return {
        "unique_material_strings": int(rows["material"].nunique()),
        "n_strings_appearing_once": int((rows["material"].value_counts() == 1).sum()),
        "n_rows_with_repeated_string": int(string_repeat.sum()),
        "pct_rows_with_repeated_string": float(100 * string_repeat.mean()),
        "same_string_different_x": int((per_string["n_x"] > 1).sum()),
        "same_string_different_composition": int((per_string["n_comp"] > 1).sum()),
        "unique_normalized_compositions": int(len(per_composition)),
        "unique_compositions_by_rounding_decimals": {
            str(d): int(len(np.unique(np.round(fractions, d), axis=0))) for d in (4, 5, 6, 8, 10)
        },
        "n_rows_with_repeated_composition": int(
            per_composition.loc[per_composition["n_rows"] > 1, "n_rows"].sum()
        ),
        "pct_rows_with_repeated_composition": float(
            100 * per_composition.loc[per_composition["n_rows"] > 1, "n_rows"].sum() / len(rows)
        ),
        "n_singleton_compositions": int((per_composition["n_rows"] == 1).sum()),
        "largest_compositions": [
            {
                "strings": sorted(member_rows(rows, "composition", c)["material"].unique())[:5],
                "n_rows": int(info["n_rows"]),
            }
            for c, info in largest.iterrows()
        ],
        "compositions_with_multiple_strings": {
            "n_compositions": int(len(multi_string)),
            "n_rows": int(multi_string["n_rows"].sum()),
            "n_strings": int(multi_string["n_strings"].sum()),
            "extra_string_groups_vs_compositions": int(
                rows["material"].nunique() - len(per_composition)
            ),
            "examples": string_examples,
        },
        "unique_x_rows_exact": int(rows["x"].nunique()),
        "unique_x_rows_within_float_noise": int(rows["x_within_noise"].nunique()),
        "same_composition_different_x": {
            "n_compositions_exact": int(len(multi_x)),
            "n_rows_exact": int(multi_x["n_rows"].sum()),
            "n_compositions_beyond_float_noise": int((spread_sd > X_NOISE_SD).sum()),
            "max_column_difference_in_sd": float(spread_sd.max()),
            "examples": x_examples,
        },
        "same_x_different_composition": {
            "n_x_rows_exact": int(len(multi_composition)),
            "n_x_rows_within_float_noise": int(
                (rows.groupby("x_within_noise")["composition"].nunique() > 1).sum()
            ),
            "examples": composition_examples,
        },
        "near_identical_x_different_composition": {
            "rounding_of_x_in_sd_units_decimals": NEAR_MISS_DECIMALS,
            "n_groups": int((near_compositions > 1).sum()),
            "examples": near_examples,
        },
    }


def check_tc_conflicts(composition, rows):
    tc = rows["tc"]
    by_composition = rows.groupby("composition")["tc"]
    stats = pd.DataFrame(
        {"n_rows": by_composition.size(), "tc_min": by_composition.min(), "tc_max": by_composition.max()}
    )
    stats["tc_range"] = stats["tc_max"] - stats["tc_min"]
    repeated = stats[stats["n_rows"] >= 2]
    tc_range = repeated["tc_range"]

    worst = repeated.sort_values(["tc_range", "n_rows"], ascending=[False, False]).head(MAX_EXAMPLES)
    worst_examples = []
    for composition_id, info in worst.iterrows():
        members = member_rows(rows, "composition", composition_id)
        worst_examples.append(
            {
                "strings": sorted(members["material"].unique())[:5],
                "n_rows": int(info["n_rows"]),
                "tc_min": round6(info["tc_min"]),
                "tc_max": round6(info["tc_max"]),
                "tc_range": round6(info["tc_range"]),
            }
        )

    ssw_by_composition = ((tc - by_composition.transform("mean")) ** 2).groupby(
        rows["composition"]
    ).sum().sort_values(ascending=False)
    cuprate = (composition["Cu"].to_numpy() > 0) & (composition["O"].to_numpy() > 0)
    groups = rows["composition"].to_numpy()
    return {
        "n_compositions_with_2plus_rows": int(len(repeated)),
        "n_rows_in_those": int(repeated["n_rows"].sum()),
        "tc_range_median": float(tc_range.median()),
        "tc_range_p90": float(np.percentile(tc_range, 90)),
        "tc_range_max": float(tc_range.max()),
        "n_range_above_10K": int((tc_range > LARGE_TC_RANGE_K).sum()),
        "n_range_exactly_0": int((tc_range == 0).sum()),
        "worst_10": worst_examples,
        "pooled_rmse_within_composition": pooled_rmse(tc, groups),
        "sensitivity": {
            "pooled_rmse_within_exact_x": pooled_rmse(tc, rows["x"].to_numpy())["rmse"],
            "pooled_rmse_within_material_string": pooled_rmse(tc, rows["material"].to_numpy())["rmse"],
            "pooled_rmse_cuprates_only": pooled_rmse(tc[cuprate], groups[cuprate])["rmse"],
            "pooled_rmse_non_cuprates": pooled_rmse(tc[~cuprate], groups[~cuprate])["rmse"],
            "cuprate_definition": "Cu > 0 and O > 0",
            "n_cuprate_rows": int(cuprate.sum()),
            "share_of_ssw_from_10_largest_contributors": float(
                ssw_by_composition.head(10).sum() / ssw_by_composition.sum()
            ),
        },
    }


def check_duplicates(X, y, material, rows):
    with_tc = X.assign(**{TARGET: y.to_numpy()})
    with_tc_and_string = with_tc.assign(material=material.to_numpy())
    composition_and_tc = rows[["composition", "tc"]]

    def duplicate_counts(frame):
        extra = int(frame.duplicated().sum())
        involved = int(frame.duplicated(keep=False).sum())
        return {"extra_rows": extra, "rows_involved": involved, "groups": involved - extra}

    return {
        "x_and_tc": duplicate_counts(with_tc),
        "x_tc_and_material_string": duplicate_counts(with_tc_and_string),
        "normalized_composition_and_tc": duplicate_counts(composition_and_tc),
    }


def compare_to_proposal(report):
    shape, tc, pca, repeats = report["1_shape"], report["2_tc"], report["4_pca"], report["5_repeats"]
    computed = {
        "rows": (shape["rows"], 0),
        "inputs": (shape["inputs"], 0),
        "element_columns": (shape["element_columns"], 0),
        "missing_values": (sum(shape["missing_values"].values()), 0),
        "tc_mean": (tc["mean"], 1),
        "tc_sd": (tc["sd_ddof1"], 1),
        "pcs_for_90pct": (pca["pcs_for_90pct"], 0),
        "unique_material_strings": (repeats["unique_material_strings"], 0),
        "pct_rows_with_repeated_string": (repeats["pct_rows_with_repeated_string"], 0),
    }
    return [
        {
            "item": item,
            "proposal": PROPOSAL[item],
            "computed": computed[item][0],
            "agrees_at_proposal_precision": bool(round(computed[item][0], computed[item][1]) == PROPOSAL[item]),
        }
        for item in PROPOSAL
    ]


def write_markdown(report, path):
    shape, tc, align, pca = report["1_shape"], report["2_tc"], report["3_alignment"], report["4_pca"]
    repeats, conflicts, dups = report["5_repeats"], report["6_tc_conflicts"], report["7_duplicates"]
    formulas = report["formula_strings"]
    multi = repeats["compositions_with_multiple_strings"]
    multi_x = repeats["same_composition_different_x"]
    pooled = conflicts["pooled_rmse_within_composition"]
    sens = conflicts["sensitivity"]
    near = repeats["near_identical_x_different_composition"]

    table = "\n".join(
        f"| {c['item']} | {c['proposal']} | "
        f"{round(c['computed'], 3) if isinstance(c['computed'], float) else c['computed']} | "
        f"{'yes' if c['agrees_at_proposal_precision'] else 'NO'} |"
        for c in report["proposal_comparison"]
    )
    near_example = ""
    if near["examples"]:
        e = near["examples"][0]
        near_example = f" Example: {e['strings'][0]} (Tc {e['tc'][0]}) vs {e['strings'][1]} (Tc {e['tc'][1]})."

    text = f"""# Data QC: superconductor Tc (UCI 464)

Produced by `src/01_data_qc.py`. Full detail and examples: `results/data_qc.json`. No rows dropped.

## Against the proposal

| Item | Proposal | Computed | Agrees (at proposal precision) |
|---|---|---|---|
{table}

## Results

- Shape: {shape['rows']} rows, {shape['inputs']} inputs, {shape['element_columns']} element columns; no missing or infinite values; no non-numeric columns in X or the element table. No constant inputs. Never-present elements: {', '.join(shape['constant_columns']['element_columns_never_present'])}.
- Tc: mean {tc['mean']:.2f}, SD {tc['sd_ddof1']:.2f}, min {tc['min']}, median {tc['median']}, max {tc['max']} ({tc['top_10'][0]['material']}); {tc['n_tc_le_0']} rows with Tc <= 0.
- Alignment: {align['n_rows_tc_differs']} rows with a Tc mismatch between the files; {align['n_rows_number_of_elements_differs']} with `number_of_elements` mismatch. Formula strings disagree with the element columns in {formulas['n_rows_string_disagrees_with_element_columns']} rows.
- PCA: {pca['pcs_for_90pct']} PCs reach {100 * pca['cumulative_variance_at_k']:.1f}% ({pca['pcs_for_90pct'] - 1} reach {100 * pca['cumulative_variance_at_k_minus_1']:.1f}%).
- Repeats: {repeats['unique_material_strings']} strings; {repeats['n_rows_with_repeated_string']} rows ({repeats['pct_rows_with_repeated_string']:.1f}%) share a string. {repeats['unique_normalized_compositions']} normalized compositions, {repeats['n_rows_with_repeated_composition']} rows ({repeats['pct_rows_with_repeated_composition']:.1f}%) share one. Same string always gives same X ({repeats['same_string_different_x']} exceptions).
- Exceptions: {multi_x['n_compositions_exact']} compositions have 2 exact X values, but differ by at most {multi_x['max_column_difference_in_sd']:.1e} SD (last-digit noise in the file); beyond that noise, {multi_x['n_compositions_beyond_float_noise']}. Same X with different composition: {repeats['same_x_different_composition']['n_x_rows_exact']}.
- Tc conflicts within repeated compositions ({conflicts['n_compositions_with_2plus_rows']} compositions, {conflicts['n_rows_in_those']} rows): range median {conflicts['tc_range_median']:.2f} K, 90th percentile {conflicts['tc_range_p90']:.1f} K, max {conflicts['tc_range_max']:.1f} K ({conflicts['worst_10'][0]['strings'][0]}); {conflicts['n_range_above_10K']} above 10 K.
- Noise floor: pooled within-composition RMSE {pooled['rmse']:.2f} K (SSW/{pooled['dof']} dof). Cuprates {sens['pooled_rmse_cuprates_only']:.2f} K, non-cuprates {sens['pooled_rmse_non_cuprates']:.2f} K; the 10 largest contributors hold {100 * sens['share_of_ssw_from_10_largest_contributors']:.0f}% of SSW, so this floor varies strongly by family.
- Exact duplicates (X and Tc): {dups['x_and_tc']['extra_rows']} extra rows in {dups['x_and_tc']['groups']} groups ({dups['x_and_tc']['rows_involved']} rows involved).

## Implications for groups and folds

1. Group by normalized composition, not by the `material` string. {multi['n_compositions']} compositions ({multi['n_rows']} rows) are written with 2+ strings (e.g. {', '.join(multi['examples'][2]['strings'])}), and their X matches to within float noise, so string groups would put the same inputs in train and test. All {dups['x_and_tc']['extra_rows']} exact (X, Tc) duplicates are spelling variants: {dups['x_tc_and_material_string']['extra_rows']} remain once the string is included.
2. Do not group on raw X values: {multi_x['n_compositions_exact']} compositions would split on last-digit noise.
3. {formulas['n_rows_bare_oxygen']} rows ({formulas['pct_rows_bare_oxygen']:.1f}%), {formulas['n_rows_bare_oxygen_with_cu']} of them with Cu, have a bare `O` stored as O = 1, most likely unspecified oxygen. Y1Ba2Cu3O and Y1Ba2Cu3O7 are then separate groups for the same compound.
4. Composition groups stop exact repeats only. Oxygen-doping siblings remain near-identical across folds.{near_example} The element-based scheme is the check for this family leakage.
"""
    path.write_text(text)


def main():
    X, y, composition, material = load_raw()
    unique_m_tc = pd.read_csv(RAW / "unique_m.csv", usecols=[TARGET])[TARGET]

    table = composition.to_numpy(dtype=float)
    fractions = table / table.sum(axis=1, keepdims=True)
    rows = pd.DataFrame(
        {
            "row": np.arange(len(X)),
            "composition": group_ids(np.round(fractions, COMPOSITION_DECIMALS)),
            "x": group_ids(X.to_numpy()),
            "x_within_noise": group_ids(
                np.round(X.to_numpy() / X.std(ddof=0).to_numpy(), X_NOISE_DECIMALS)
            ),
            "material": material.to_numpy(),
            "tc": y.to_numpy(),
        }
    )

    report = {
        "1_shape": check_shape(X, y, composition, material),
        "2_tc": check_tc(y, material),
        "3_alignment": check_alignment(X, y, composition, material, unique_m_tc),
        "4_pca": check_pca(X),
        "5_repeats": check_repeats(X, composition, fractions, rows),
        "6_tc_conflicts": check_tc_conflicts(composition, rows),
        "7_duplicates": check_duplicates(X, y, material, rows),
        "formula_strings": check_formula_strings(composition, material),
    }
    report["proposal_comparison"] = compare_to_proposal(report)

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "data_qc.json").write_text(json.dumps(report, indent=2) + "\n")
    write_markdown(report, RESULTS / "data_qc.md")
    print(f"Wrote {RESULTS / 'data_qc.json'}")
    print(f"Wrote {RESULTS / 'data_qc.md'}")


if __name__ == "__main__":
    main()
