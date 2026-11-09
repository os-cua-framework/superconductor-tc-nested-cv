# Nested CV summary

375 job records read. Finished jobs per model (of 75): mean 75, ols 75, ridge 75, lasso 75, rf 75.

## Test RMSE by model and scheme (K)

Only seeds with all 5 folds finished are used. rmse_sd and rmse_se are computed
within each seed and then averaged over seeds; seed_sd is the SD of the seed means.

| model | scheme | n_seeds | n_folds | rmse_mean | rmse_sd | rmse_se | seed_sd | r2_mean |
|---|---|---|---|---|---|---|---|---|
| mean | random | 5 | 25 | 34.255 | 0.225 | 0.101 | 0.002 | -0.000 |
| mean | formula | 5 | 25 | 34.262 | 0.828 | 0.370 | 0.008 | -0.003 |
| mean | element | 5 | 25 | 34.312 | 1.418 | 0.634 | 0.043 | -0.014 |
| ols | random | 5 | 25 | 17.625 | 0.192 | 0.086 | 0.004 | 0.735 |
| ols | formula | 5 | 25 | 17.733 | 0.254 | 0.114 | 0.017 | 0.731 |
| ols | element | 5 | 25 | 18.199 | 0.644 | 0.288 | 0.068 | 0.714 |
| ridge | random | 5 | 25 | 17.625 | 0.192 | 0.086 | 0.004 | 0.735 |
| ridge | formula | 5 | 25 | 17.734 | 0.253 | 0.113 | 0.017 | 0.731 |
| ridge | element | 5 | 25 | 18.217 | 0.631 | 0.282 | 0.058 | 0.713 |
| lasso | random | 5 | 25 | 17.625 | 0.192 | 0.086 | 0.004 | 0.735 |
| lasso | formula | 5 | 25 | 17.735 | 0.253 | 0.113 | 0.017 | 0.731 |
| lasso | element | 5 | 25 | 18.207 | 0.630 | 0.282 | 0.064 | 0.713 |
| rf | random | 5 | 25 | 9.292 | 0.249 | 0.111 | 0.042 | 0.926 |
| rf | formula | 5 | 25 | 9.596 | 0.403 | 0.180 | 0.022 | 0.921 |
| rf | element | 5 | 25 | 13.419 | 1.007 | 0.450 | 0.089 | 0.844 |

## Paired gap: best linear model minus random forest

A positive gap means the forest has the lower RMSE. Pairs are the same (seed, fold).
tie is true when the absolute mean gap is within one SD of the fold gaps. best_linear
is picked on the same outer-test RMSE used for the gap, which slightly favours it.

| scheme | best_linear | n_folds | seeds_used | n_complete_seeds | gap_mean | gap_sd | gap_se | tie |
|---|---|---|---|---|---|---|---|---|
| random | ols | 25 | 0;1;2;3;4 | 5 | 8.333 | 0.270 | 0.130 | False |
| formula | ols | 25 | 0;1;2;3;4 | 5 | 8.137 | 0.364 | 0.171 | False |
| element | ols | 25 | 0;1;2;3;4 | 5 | 4.781 | 0.795 | 0.364 | False |

## Change in RMSE when folds are grouped (K)

Positive means RMSE rises compared with random folds.

| model | rmse_mean_random | rmse_mean_formula | rmse_mean_element | change_random_to_formula | change_random_to_element | n_seeds_random | n_seeds_formula | n_seeds_element |
|---|---|---|---|---|---|---|---|---|
| mean | 34.255 | 34.262 | 34.312 | 0.007 | 0.057 | 5 | 5 | 5 |
| ols | 17.625 | 17.733 | 18.199 | 0.108 | 0.575 | 5 | 5 | 5 |
| ridge | 17.625 | 17.734 | 18.217 | 0.110 | 0.592 | 5 | 5 | 5 |
| lasso | 17.625 | 17.735 | 18.207 | 0.110 | 0.582 | 5 | 5 | 5 |
| rf | 9.292 | 9.596 | 13.419 | 0.305 | 4.127 | 5 | 5 | 5 |

## Selected settings

| model | scheme | folds | penalty min | median | max | at grid edge | at low edge | at high edge | most common setting (count) |
|---|---|---|---|---|---|---|---|---|---|
| ridge | random | 25 | 0.0149 | 0.0356 | 0.0853 | 0 | 0 | 0 | alpha=0.0356 (13) |
| ridge | formula | 25 | 1e-06 | 0.0356 | 0.204 | 7 | 7 | 0 | alpha=1e-06 (7) |
| ridge | element | 25 | 1e-06 | 2.81 | 6.72 | 2 | 2 | 0 | alpha=6.72 (10) |
| lasso | random | 25 | 8.53e-06 | 5.74e-05 | 0.000386 | 0 | 0 | 0 | alpha=5.74e-05 (11) |
| lasso | formula | 25 | 1e-07 | 1e-07 | 0.000728 | 18 | 18 | 0 | alpha=1e-07 (18) |
| lasso | element | 25 | 1e-07 | 0.00259 | 0.00924 | 2 | 2 | 0 | alpha=0.00259 (7) |
| rf | random | 25 | - | - | - | 0 | 0 | 0 | max_depth=20, max_features=0.2, max_samples=None, min_samples_leaf=1 (23) |
| rf | formula | 25 | - | - | - | 0 | 0 | 0 | max_depth=20, max_features=0.2, max_samples=None, min_samples_leaf=1 (18) |
| rf | element | 25 | - | - | - | 0 | 0 | 0 | max_depth=15, max_features=0.75, max_samples=None, min_samples_leaf=3 (7) |

Convergence warnings, total per model: mean 0, ols 0, ridge 0, lasso 0, rf 0.

## Notes

- ridge chose the smallest penalty in 9 of 75 folds and the largest in 0; the best penalty for those folds may lie outside the grid.
- lasso chose the smallest penalty in 20 of 75 folds and the largest in 0; the best penalty for those folds may lie outside the grid.
- For rf, at_grid_edge only says the chosen setting was the first or last of the random draws. Those draws are unordered, so it does not mean a search boundary.
- No problems found while reading the job records.
