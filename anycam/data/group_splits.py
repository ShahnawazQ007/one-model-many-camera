"""Grouped K-fold for the fruit (group = fruit) and leaf (group = leaf) datasets."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def grouped_cv(df: pd.DataFrame, group_col: str, strat_col: str, n_folds: int = 5, seeds=(0, 1, 2),
               val_frac: float = 0.15):
    groups = df.groupby(group_col)[strat_col].first()
    out = []
    for seed in seeds:
        sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (tr, te) in enumerate(sgkf.split(groups.index, groups.values, groups=groups.index)):
            tr_g = list(groups.index[tr])
            rng = np.random.default_rng(1000 * seed + fold)
            n_val = max(1, int(round(val_frac * len(tr_g))))
            val_g = list(rng.choice(tr_g, n_val, replace=False))
            tr_g = [g for g in tr_g if g not in set(val_g)]
            rows = lambda gs: df.index[df[group_col].isin(gs)].tolist()
            out.append({"seed": seed, "fold": fold, "train": rows(tr_g), "val": rows(val_g),
                        "test": rows(list(groups.index[te]))})
    return out
