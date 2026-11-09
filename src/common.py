"""Shared paths, data loading and the seeded group K-fold used by every script."""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
JOBS = RESULTS / "jobs"
FIGURES = ROOT / "figures"

TARGET = "critical_temp"
N_FOLDS = 5
SEEDS = [0, 1, 2, 3, 4]
SCHEMES = ["random", "formula", "element"]

GROUPS_FILE = PROCESSED / "groups.csv.gz"
SPLITS_FILE = PROCESSED / "outer_splits.csv.gz"


def load_raw():
    """Return inputs X, target y, the element composition table and the formula strings."""
    train = pd.read_csv(RAW / "train.csv")
    formulas = pd.read_csv(RAW / "unique_m.csv")
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    composition = formulas.drop(columns=[TARGET, "material"])
    return X, y, composition, formulas["material"]


def seeded_group_folds(groups, n_splits, seed):
    """Return a fold id per row, keeping every group whole.

    Groups are shuffled by the seed and each one goes to the currently smallest
    fold. sklearn's deterministic GroupKFold is not used because it places large
    groups the same way for every seed.
    """
    unique, inverse, counts = np.unique(
        np.asarray(groups), return_inverse=True, return_counts=True
    )
    order = np.random.RandomState(seed).permutation(len(unique))
    fold_sizes = np.zeros(n_splits, dtype=int)
    group_fold = np.empty(len(unique), dtype=int)
    for g in order:
        k = int(np.argmin(fold_sizes))
        group_fold[g] = k
        fold_sizes[k] += counts[g]
    return group_fold[inverse]


def split_column(scheme, seed):
    """Name of the fold-id column for one outer split in SPLITS_FILE."""
    return f"{scheme}_s{seed}"


def load_splits():
    """Return the frozen outer fold ids, one column per scheme and seed."""
    return pd.read_csv(SPLITS_FILE)


def load_groups():
    """Return the integer group ids per row (formula_group, element_group)."""
    return pd.read_csv(GROUPS_FILE)
