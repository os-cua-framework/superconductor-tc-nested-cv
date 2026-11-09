# Split report

N = 21263, 5 folds, seeds [0, 1, 2, 3, 4].
All checks in src/02_make_splits.py passed.

## Groups

| grouping | groups | singletons | largest | top 1 | top 5 | top 10 |
|---|---|---|---|---|---|---|
| formula_group | 15164 | 12742 | 110 | 0.5% | 1.6% | 2.5% |
| element_group | 3365 | 1169 | 720 | 3.4% | 9.0% | 12.8% |

## Formula groups versus formula strings

15542 unique strings -> 15309 raw composition vectors (-233: same amounts, different text; 210 element reorderings, 23 notation variants) -> 15164 normalized compositions (-145: scaled copies of one formula) -> 15164 formula groups (-0: exact-input link).

There are 15170 unique input rows, slightly more than the groups: 6 compositions have inputs that differ only by floating-point noise.

351 formula groups (1508 rows) contain more than one string, so folds grouped by string could split identical inputs.

## Largest element sets

| elements | rows | mean Tc (K) |
|---|---|---|
| Ba-Cu-O-Y | 720 | 75.39 |
| Cu-La-O-Sr | 358 | 28.76 |
| Bi-Ca-Cu-O-Sr | 323 | 81.21 |
| Ba-Cu-O-Y-Zn | 267 | 50.48 |
| Ba-Cu-O-Pr-Y | 254 | 53.85 |
| Ba-Cu-La-O-Y | 167 | 71.81 |
| Ba-Cu-Fe-O-Y | 166 | 63.77 |
| Ba-Cu-La-O | 155 | 27.66 |
| Ba-Ca-Cu-La-O | 154 | 55.25 |
| Ba-Cu-Hg-O | 150 | 78.21 |

Distinct element sets: 3365.

## Folds

| split | fold sizes | largest/smallest | min fold mean Tc | max fold mean Tc |
|---|---|---|---|---|
| random_s0 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 34.15 | 34.73 |
| random_s1 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 33.52 | 34.92 |
| random_s2 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 33.70 | 35.46 |
| random_s3 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 33.73 | 34.90 |
| random_s4 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 33.13 | 35.35 |
| formula_s0 | 4250, 4250, 4250, 4249, 4264 | 1.004 | 33.48 | 35.67 |
| formula_s1 | 4253, 4253, 4252, 4253, 4252 | 1.000 | 32.87 | 36.18 |
| formula_s2 | 4253, 4253, 4253, 4252, 4252 | 1.000 | 32.24 | 36.71 |
| formula_s3 | 4270, 4248, 4248, 4248, 4249 | 1.005 | 32.46 | 36.09 |
| formula_s4 | 4253, 4254, 4252, 4252, 4252 | 1.000 | 32.50 | 38.44 |
| element_s0 | 4276, 4244, 4240, 4259, 4244 | 1.008 | 30.10 | 41.45 |
| element_s1 | 4277, 4252, 4248, 4241, 4245 | 1.008 | 30.85 | 37.09 |
| element_s2 | 4207, 4209, 4335, 4305, 4207 | 1.030 | 31.57 | 40.38 |
| element_s3 | 4245, 4245, 4251, 4276, 4246 | 1.007 | 30.61 | 38.60 |
| element_s4 | 4250, 4264, 4245, 4256, 4248 | 1.004 | 30.43 | 40.60 |

## Agreement between seeds (adjusted Rand index)

| scheme | min | mean | max | expected if independent |
|---|---|---|---|---|
| formula | 0.0003 | 0.0007 | 0.0011 | 0.0008 |
| element | 0.0095 | 0.0155 | 0.0222 | 0.0129 |

Pairwise values for every seed pair are in split_report.json.

## Random folds

Under `random`, 36.9% to 37.2% of rows have an exact-input copy on the other side of their fold.

## Flags

No split has largest/smallest fold above 1.3.

## SHA-256

- `groups.csv.gz`: `d73ca60c1b3c9fb8ee051d2f9dc08baf036eed229d592fe94a4f7c782660946c`
- `outer_splits.csv.gz`: `84c2bf07582d49f9d6f6f8529814747cd087f41ef503f85df3092dcf865d31cf`
