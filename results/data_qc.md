# Data QC: superconductor Tc (UCI 464)

Produced by `src/01_data_qc.py`. Full detail and examples: `results/data_qc.json`. No rows dropped.

## Against the proposal

| Item | Proposal | Computed | Agrees (at proposal precision) |
|---|---|---|---|
| rows | 21263 | 21263 | yes |
| inputs | 81 | 81 | yes |
| element_columns | 86 | 86 | yes |
| missing_values | 0 | 0 | yes |
| tc_mean | 34.4 | 34.421 | yes |
| tc_sd | 34.3 | 34.254 | yes |
| pcs_for_90pct | 12 | 12 | yes |
| unique_material_strings | 15542 | 15542 | yes |
| pct_rows_with_repeated_string | 37 | 37.539 | NO |

## Results

- Shape: 21263 rows, 81 inputs, 86 element columns; no missing or infinite values; no non-numeric columns in X or the element table. No constant inputs. Never-present elements: He, Ne, Ar, Kr, Xe, Pm, Po, At, Rn.
- Tc: mean 34.42, SD 34.25, min 0.00021, median 20.0, max 185.0 (H2S1); 0 rows with Tc <= 0.
- Alignment: 0 rows with a Tc mismatch between the files; 0 with `number_of_elements` mismatch. Formula strings disagree with the element columns in 0 rows.
- PCA: 12 PCs reach 90.9% (11 reach 89.4%).
- Repeats: 15542 strings; 7982 rows (37.5%) share a string. 15164 normalized compositions, 8521 rows (40.1%) share one. Same string always gives same X (0 exceptions).
- Exceptions: 6 compositions have 2 exact X values, but differ by at most 8.9e-15 SD (last-digit noise in the file); beyond that noise, 0. Same X with different composition: 0.
- Tc conflicts within repeated compositions (2422 compositions, 8521 rows): range median 2.19 K, 90th percentile 21.3 K, max 125.0 K (H2S1); 562 above 10 K.
- Noise floor: pooled within-composition RMSE 8.03 K (SSW/6099 dof). Cuprates 10.70 K, non-cuprates 3.95 K; the 10 largest contributors hold 31% of SSW, so this floor varies strongly by family.
- Exact duplicates (X and Tc): 66 extra rows in 65 groups (131 rows involved).

## Implications for groups and folds

1. Group by normalized composition, not by the `material` string. 351 compositions (1508 rows) are written with 2+ strings (e.g. Al0.25Nb0.75, Al1Nb3, Al25Nb75), and their X matches to within float noise, so string groups would put the same inputs in train and test. All 66 exact (X, Tc) duplicates are spelling variants: 0 remain once the string is included.
2. Do not group on raw X values: 6 compositions would split on last-digit noise.
3. 1927 rows (9.1%), 1901 of them with Cu, have a bare `O` stored as O = 1, most likely unspecified oxygen. Y1Ba2Cu3O and Y1Ba2Cu3O7 are then separate groups for the same compound.
4. Composition groups stop exact repeats only. Oxygen-doping siblings remain near-identical across folds. Example: Y1Ba2Cu3O6.829 (Tc 57.7) vs Y1Ba2Cu3O6.828 (Tc 88.0). The element-based scheme is the check for this family leakage.
