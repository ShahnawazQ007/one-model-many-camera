"""HyperLeaf2024: wheat flag leaves, Specim IQ, 204 bands, 48 x 352 px.

Reads images straight from HyperLeaf2024.zip (or an unzipped folder).
Labels: train.csv (1,590) + solution.csv (820 test answers) = 2,410 leaves.
Plot IDs are not given; they are recovered from GrainWeight, which is measured
per plot: 36 unique values = 36 plots (4 cultivars x 3 fertilizer levels x 3 repeats).
"""
from __future__ import annotations

import io
import pathlib
import zipfile
from typing import Optional

import numpy as np
import pandas as pd

CULTIVARS = ["Heerup", "Kvium", "Rembrandt", "Sheriff"]
REGRESSION_TARGETS = ["GrainWeight", "Gsw", "PhiPS2", "Fertilizer"]
# Specim IQ: 204 bands, ~397-1004 nm. ASSUMED (no wavelength list ships with the data).
WAVELENGTHS = np.linspace(397.32, 1003.58, 204)
# Stored as uint16; assumed reflectance x 10,000 (mean leaf value ~0.27). ASSUMED.
SCALE = 1.0 / 10000.0


def load_labels(root) -> pd.DataFrame:
    """One row per leaf: ImageId, traits, cultivar (name + index), plot_id, official_split."""
    root = pathlib.Path(root)
    tr = _read_csv(root, "train.csv").assign(official_split="train")
    te = _read_csv(root, "solution.csv").drop(columns=["Usage"], errors="ignore").assign(official_split="test")
    df = pd.concat([tr, te], ignore_index=True)
    onehot = df[CULTIVARS].to_numpy()
    if not (onehot.sum(1) == 1).all():
        raise ValueError("cultivar one-hot columns do not sum to 1 for every leaf")
    df["cultivar_idx"] = onehot.argmax(1)
    df["cultivar"] = [CULTIVARS[i] for i in df["cultivar_idx"]]
    df["treatment"] = df["cultivar"] + "_F" + df["Fertilizer"].map(lambda f: f"{f:g}")
    # plot = unique (cultivar, fertilizer, grain weight); numbered in a stable order
    key = df[["cultivar", "Fertilizer", "GrainWeight"]].apply(tuple, axis=1)
    order = {k: i for i, k in enumerate(sorted(set(key)))}
    df["plot_id"] = key.map(order).astype(int)
    df = df.sort_values("ImageId").reset_index(drop=True)
    return df


def check_plots(df: pd.DataFrame) -> dict:
    """Sanity checks on the recovered plots; raises if the structure is not as expected."""
    n_plots = df.plot_id.nunique()
    per_treat = df.groupby("treatment").plot_id.nunique()
    shared = df.groupby("plot_id").official_split.nunique()
    out = {"leaves": len(df), "plots": n_plots, "plots_per_treatment": per_treat.to_dict(),
           "plots_in_both_official_splits": int((shared > 1).sum()),
           "leaves_per_plot_min": int(df.plot_id.value_counts().min()),
           "leaves_per_plot_max": int(df.plot_id.value_counts().max())}
    if n_plots != 36 or (per_treat != 3).any():
        raise ValueError(f"unexpected plot structure: {out}")
    return out


def _read_csv(root: pathlib.Path, name: str) -> pd.DataFrame:
    if root.is_file() and root.suffix == ".zip":
        with zipfile.ZipFile(root) as z:
            return pd.read_csv(io.BytesIO(z.read(name)))
    return pd.read_csv(root / name)


class HyperLeafImages:
    """Reads one leaf cube by ImageId -> float32 (204, 48, 352) reflectance.

    `root` = HyperLeaf2024.zip or the unzipped folder. The zip handle is opened lazily
    per process, so it is safe with DataLoader workers.
    """

    def __init__(self, root, scale: float = SCALE):
        self.root = pathlib.Path(root)
        self.scale = scale
        self._zip: Optional[zipfile.ZipFile] = None
        self._zip_pid = None

    def _bytes(self, name: str) -> bytes:
        if self.root.suffix == ".zip":
            import os
            if self._zip is None or self._zip_pid != os.getpid():
                self._zip = zipfile.ZipFile(self.root)
                self._zip_pid = os.getpid()
            return self._zip.read(name)
        return (self.root / name).read_bytes()

    def __call__(self, image_id: int) -> np.ndarray:
        import tifffile
        a = tifffile.imread(io.BytesIO(self._bytes(f"images/{int(image_id):05d}.tiff")))
        if a.shape[0] != 204:
            raise ValueError(f"image {image_id}: unexpected shape {a.shape}")
        return a.astype(np.float32) * self.scale


class HyperLeafDataset:
    """Indexable dataset (works with torch DataLoader without importing torch here).

    item -> dict(cube=(204,H,W) float32, cultivar=int, targets=float32[4], image_id, plot_id)
    `transform(cube)` is applied to the cube (e.g. a camera simulator); regression
    targets can be standardised with `target_stats` = (mean[4], std[4]) from TRAIN rows only.
    """

    def __init__(self, labels: pd.DataFrame, images: HyperLeafImages, rows=None,
                 transform=None, target_stats=None):
        self.df = labels if rows is None else labels.iloc[list(rows)].reset_index(drop=True)
        self.images = images
        self.transform = transform
        self.mean, self.std = (np.zeros(4), np.ones(4)) if target_stats is None else target_stats

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        cube = self.images(r.ImageId)
        if self.transform is not None:
            cube = self.transform(cube)
        t = (r[REGRESSION_TARGETS].to_numpy(float) - self.mean) / self.std
        return {"cube": cube, "cultivar": int(r.cultivar_idx), "targets": t.astype(np.float32),
                "image_id": int(r.ImageId), "plot_id": int(r.plot_id)}


def target_stats(labels: pd.DataFrame, rows) -> tuple:
    """Mean and std of the regression targets over the given (training) rows."""
    v = labels.iloc[list(rows)][REGRESSION_TARGETS].to_numpy(float)
    return v.mean(0), v.std(0) + 1e-8
