# How much do tree ensembles beat linear models on superconductor Tc?

Course project code. It compares the test RMSE of linear models and a random forest for predicting a superconductor's critical temperature under nested 5-fold cross-validation, with three kinds of outer split:

- `random`: ordinary shuffled folds
- `formula`: rows with the same normalized composition stay in one fold
- `element`: rows with the same set of elements stay in one fold

Each kind of split is repeated over seeds 0 to 4. All models use the same frozen folds in `data/processed/outer_splits.csv.gz`.

## Data

UCI Superconductivity Data, Hamidieh (2018), <https://archive.ics.uci.edu/dataset/464/superconductivty+data>, CC BY 4.0. The raw files are not stored here; `src/00_get_data.sh` downloads them and checks the archive's SHA-256.

## Reproduce

Python 3.12 with the versions in `requirements.txt`. Run from the repository root:

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
sh src/00_get_data.sh
.venv/bin/python src/01_data_qc.py
.venv/bin/python src/02_make_splits.py
.venv/bin/python src/03_nested_cv.py --models mean ols ridge lasso
.venv/bin/python src/03_nested_cv.py --models rf --rf-trees 100
.venv/bin/python src/04_summarize.py
.venv/bin/python src/05_figure.py
```

`03_nested_cv.py` saves one record per model, split, seed and outer fold in `results/jobs`, with the test predictions in `results/preds`. A rerun skips jobs whose saved record matches the current settings and splits. The random forest takes several hours on 10 cores.

## Where things are

| Path | Contents |
|---|---|
| `src/common.py` | paths, data loading, seeded group K-fold |
| `src/03_nested_cv.py` | models, tuning grids, nested CV |
| `results/summary.md` | readable summary of results |
| `results/data_qc.md`, `results/split_report.md` | data checks and split checks |
| `figures/baseline.png` | RMSE by model and split type |
