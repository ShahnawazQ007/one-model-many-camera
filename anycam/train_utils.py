"""Shared training pieces: band normalisation, augmentation, metrics."""
from __future__ import annotations

import numpy as np


class BandNorm:
    """Per-band standardisation from TRAINING leaf pixels; background (all-zero) stays 0."""

    def __init__(self, mean, std):
        self.mean = np.asarray(mean, np.float32)[:, None, None]
        self.std = np.asarray(std, np.float32)[:, None, None]

    @classmethod
    def fit(cls, cubes):
        """cubes: iterable of (C,H,W) arrays (training images only)."""
        s = s2 = n = 0
        for c in cubes:
            m = c.sum(0) > 0
            px = c[:, m].astype(np.float64)
            s = s + px.sum(1); s2 = s2 + (px ** 2).sum(1); n += px.shape[1]
        mean = s / n
        std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
        return cls(mean, std)

    def __call__(self, cube):
        m = (cube.sum(0) > 0)[None]
        return ((cube - self.mean) / self.std * m).astype(np.float32)

    def state(self):
        return {"mean": self.mean.ravel().tolist(), "std": self.std.ravel().tolist()}


def random_flip(cube, rng):
    if rng.random() < 0.5:
        cube = cube[:, :, ::-1]
    if rng.random() < 0.5:
        cube = cube[:, ::-1, :]
    return np.ascontiguousarray(cube)


def classification_metrics(y_true, y_pred, n_classes):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    acc = float((y_true == y_pred).mean())
    f1s = []
    for k in sorted(set(y_true.tolist()) | set(y_pred.tolist())):  # classes present
        tp = np.sum((y_pred == k) & (y_true == k)); fp = np.sum((y_pred == k) & (y_true != k)); fn = np.sum((y_pred != k) & (y_true == k))
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return {"accuracy": acc, "macro_f1": float(np.mean(f1s))}


def r2_rmse(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    ss = np.sum((y_true - y_true.mean()) ** 2)
    return {"R2": float(1 - np.sum((y_true - y_pred) ** 2) / ss) if ss > 0 else float("nan"),
            "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2)))}
