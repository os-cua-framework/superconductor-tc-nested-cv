"""Nested cross-validation on the frozen outer splits.

One job is one (model, scheme, seed, outer fold). A job tunes on the outer
training rows only, refits the chosen setting on all of them, and scores the
outer test rows. Scaling sits inside the model pipeline, so it is fit on the
training part of every inner and outer split. Inner folds use the same grouping
as the outer split. Finished jobs are saved in results/jobs and skipped on a
rerun, so a stopped run resumes where it left off.
"""
import argparse
import hashlib
import json
import os
import time
import warnings

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[var] = "1"

import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, ParameterSampler
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    GROUPS_FILE,
    JOBS,
    N_FOLDS,
    RESULTS,
    SCHEMES,
    SEEDS,
    SPLITS_FILE,
    load_groups,
    load_raw,
    load_splits,
    seeded_group_folds,
    split_column,
)

PREDS = RESULTS / "preds"

RIDGE_ALPHAS = np.logspace(-6, 5, 30)
LASSO_ALPHAS = np.logspace(-7, 1, 30)
N_RF_SETTINGS = 30
RF_SPACE = {
    "max_features": [0.1, 0.2, 1 / 3, 0.5, 0.75, 1.0],
    "min_samples_leaf": [1, 2, 3, 5, 10],
    "max_depth": [None, 10, 15, 20, 30],
    "max_samples": [0.5, 0.75, None],
}


def build_model(name, rf_trees):
    """Return the estimator and its list of settings to compare by inner CV."""
    if name == "mean":
        return DummyRegressor(strategy="mean"), [{}]
    if name == "ols":
        return make_pipeline(StandardScaler(), LinearRegression()), [{}]
    if name == "ridge":
        settings = [{"ridge__alpha": float(a)} for a in RIDGE_ALPHAS]
        return make_pipeline(StandardScaler(), Ridge()), settings
    if name == "lasso":
        settings = [{"lasso__alpha": float(a)} for a in LASSO_ALPHAS]
        model = make_pipeline(
            StandardScaler(), Lasso(max_iter=200000, tol=1e-4, precompute=True)
        )
        return model, settings
    if name == "rf":
        settings = list(ParameterSampler(RF_SPACE, n_iter=N_RF_SETTINGS, random_state=0))
        return RandomForestRegressor(n_estimators=rf_trees, n_jobs=1), settings
    raise ValueError(f"unknown model: {name}")


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def inner_fold_ids(scheme, seed, fold, train, groups):
    """Return inner fold ids over the outer training rows, grouped like the outer split."""
    inner_seed = 1000 * (seed + 1) + fold
    if scheme == "random":
        ids = np.empty(len(train), dtype=int)
        splitter = KFold(N_FOLDS, shuffle=True, random_state=inner_seed)
        for k, (_, val) in enumerate(splitter.split(train)):
            ids[val] = k
        return ids
    return seeded_group_folds(groups[train], N_FOLDS, inner_seed)


def inner_rmse(model, setting, X, y, inner_ids):
    """Return the mean validation RMSE of one setting over the inner folds."""
    errors = []
    for k in range(N_FOLDS):
        val = inner_ids == k
        fitted = clone(model).set_params(**setting).fit(X[~val], y[~val])
        errors.append(rmse(y[val], fitted.predict(X[val])))
    return float(np.mean(errors))


def job_name(model, scheme, seed, fold):
    return f"{model}__{scheme}__s{seed}__f{fold}"


def job_config(name, rf_trees, data_hash):
    """Return a short hash of everything that defines a job's result."""
    _, settings = build_model(name, rf_trees)
    payload = {
        "model": name,
        "settings": settings,
        "rf_trees": rf_trees if name == "rf" else None,
        "data": data_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def job_is_done(model, scheme, seed, fold, config):
    """A job counts as done only with a valid record for this config and its predictions."""
    base = job_name(model, scheme, seed, fold)
    record = JOBS / f"{base}.json"
    if not record.exists() or not (PREDS / f"{base}.npz").exists():
        return False
    try:
        return json.loads(record.read_text()).get("config") == config
    except json.JSONDecodeError:
        return False


def run_job(name, scheme, seed, fold, X, y, groups, outer, rf_trees, config):
    """Tune, refit and score one outer fold, then save the record and predictions."""
    start = time.time()
    train = np.flatnonzero(outer != fold)
    test = np.flatnonzero(outer == fold)
    model, settings = build_model(name, rf_trees)
    if "random_state" in model.get_params():
        model.set_params(random_state=100 * seed + fold)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        inner_scores = []
        best = 0
        if len(settings) > 1:
            inner_ids = inner_fold_ids(scheme, seed, fold, train, groups)
            inner_scores = [
                inner_rmse(model, s, X[train], y[train], inner_ids) for s in settings
            ]
            best = int(np.argmin(inner_scores))
        final = clone(model).set_params(**settings[best]).fit(X[train], y[train])
        pred = final.predict(X[test])
        train_pred = final.predict(X[train])

    record = {
        "config": config,
        "model": name,
        "scheme": scheme,
        "seed": seed,
        "fold": fold,
        "n_train": len(train),
        "n_test": len(test),
        "rmse": rmse(y[test], pred),
        "r2": float(r2_score(y[test], pred)),
        "train_rmse": rmse(y[train], train_pred),
        "n_settings": len(settings),
        "best_index": best,
        "best_setting": settings[best],
        "at_grid_edge": bool(name in ("ridge", "lasso") and best in (0, len(settings) - 1)),
        "inner_rmse": inner_scores,
        "rf_trees": rf_trees if name == "rf" else None,
        "convergence_warnings": sum(issubclass(w.category, ConvergenceWarning) for w in caught),
        "seconds": time.time() - start,
    }

    base = job_name(name, scheme, seed, fold)
    tmp_preds = PREDS / f"{base}.tmp.npz"
    np.savez_compressed(tmp_preds, row_id=test, pred=pred)
    os.replace(tmp_preds, PREDS / f"{base}.npz")
    tmp_job = JOBS / f"{base}.tmp"
    tmp_job.write_text(json.dumps(record, indent=1))
    os.replace(tmp_job, JOBS / f"{base}.json")
    print(f"done {base} rmse={record['rmse']:.3f} in {record['seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=["mean", "ols", "ridge", "lasso"])
    parser.add_argument("--schemes", nargs="+", default=SCHEMES)
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--folds", nargs="+", type=int, default=list(range(N_FOLDS)))
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--rf-trees", type=int, default=100)
    args = parser.parse_args()

    JOBS.mkdir(parents=True, exist_ok=True)
    PREDS.mkdir(parents=True, exist_ok=True)

    X_frame, y_series, _, _ = load_raw()
    X = X_frame.to_numpy(dtype=float)
    y = y_series.to_numpy(dtype=float)
    splits = load_splits()
    group_table = load_groups()
    assert (splits["row_id"].to_numpy() == np.arange(len(y))).all()
    assert (group_table["row_id"].to_numpy() == np.arange(len(y))).all()
    groups = {
        "random": None,
        "formula": group_table["formula_group"].to_numpy(),
        "element": group_table["element_group"].to_numpy(),
    }

    data_hash = hashlib.sha256(SPLITS_FILE.read_bytes() + GROUPS_FILE.read_bytes()).hexdigest()
    configs = {model: job_config(model, args.rf_trees, data_hash) for model in args.models}
    jobs = [
        (model, scheme, seed, fold)
        for seed in args.seeds
        for scheme in args.schemes
        for model in args.models
        for fold in args.folds
        if not job_is_done(model, scheme, seed, fold, configs[model])
    ]
    print(f"{len(jobs)} jobs to run on {args.workers} workers", flush=True)
    Parallel(n_jobs=args.workers, pre_dispatch="n_jobs")(
        delayed(run_job)(
            model, scheme, seed, fold, X, y, groups[scheme],
            splits[split_column(scheme, seed)].to_numpy(), args.rf_trees, configs[model],
        )
        for model, scheme, seed, fold in jobs
    )
    print("all jobs finished", flush=True)


if __name__ == "__main__":
    main()
