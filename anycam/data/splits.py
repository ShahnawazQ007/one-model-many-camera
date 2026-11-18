"""Leakage-free splits.

HyperLeaf: 5-fold cross-validation grouped by PLOT and stratified by cultivar
(9 plots per cultivar; 12 treatments x 3 plots is too few to stratify 5 ways), repeated with 3 seeds. Inside each training part, 3 whole
plots (one per fertilizer level where possible) are held out for validation / early stopping.
Fruit and leaf datasets: grouped by fruit or leaf.
"""
from __future__ import annotations

import json
import pathlib
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def plot_grouped_cv(df: pd.DataFrame, n_folds: int = 5, seeds=(0, 1, 2), n_val_plots: int = 3) -> List[Dict]:
    """List of splits: {seed, fold, train, val, test} with ROW indices into df."""
    plots = df.groupby("plot_id").agg(cultivar=("cultivar", "first"), fert=("Fertilizer", "first"))
    out = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        # split at plot level (one row per plot) so the stratification is over plots
        for fold, (tr_p, te_p) in enumerate(sgkf.split(plots.index, plots.cultivar, groups=plots.index)):
            train_plots = plots.index[tr_p]
            test_plots = plots.index[te_p]
            rng = np.random.default_rng(1000 * seed + fold)
            val_plots = []
            for fert in sorted(plots.loc[train_plots, "fert"].unique()):
                cand = [p for p in train_plots if plots.loc[p, "fert"] == fert and p not in val_plots]
                if cand and len(val_plots) < n_val_plots:
                    val_plots.append(int(rng.choice(cand)))
            train_plots = [int(p) for p in train_plots if p not in val_plots]
            rows = lambda ps: df.index[df.plot_id.isin(ps)].tolist()
            out.append({"seed": seed, "fold": fold,
                        "train": rows(train_plots), "val": rows(val_plots), "test": rows(test_plots),
                        "train_plots": train_plots, "val_plots": val_plots, "test_plots": [int(p) for p in test_plots]})
    return out


def official_split(df: pd.DataFrame, n_val_plots: int = 3, seed: int = 0) -> Dict:
    """The competition split (train.csv vs solution.csv), with validation plots taken from train."""
    tr_plots = sorted(df.loc[df.official_split == "train", "plot_id"].unique())
    rng = np.random.default_rng(seed)
    val_plots = [int(p) for p in rng.choice(tr_plots, n_val_plots, replace=False)]
    tr = [int(p) for p in tr_plots if p not in val_plots]
    te = sorted(int(p) for p in df.loc[df.official_split == "test", "plot_id"].unique())
    rows = lambda ps: df.index[df.plot_id.isin(ps)].tolist()
    return {"seed": seed, "fold": "official", "train": rows(tr), "val": rows(val_plots), "test": rows(te),
            "train_plots": tr, "val_plots": val_plots, "test_plots": te}


def assert_no_leakage(df: pd.DataFrame, split: Dict, group_col: str = "plot_id") -> None:
    g = {k: set(df.loc[split[k], group_col]) for k in ("train", "val", "test")}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = g[a] & g[b]
        if shared:
            raise AssertionError(f"{group_col} shared between {a} and {b}: {sorted(shared)[:5]}")


def save_splits(splits, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits))


def load_splits(path):
    return json.loads(pathlib.Path(path).read_text())
