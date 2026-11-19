"""Camera transforms used in training and testing (Step 6 onwards).

SimulatedCamera turns a raw HyperLeaf cube (204, H, W) into what a real camera would record
(B, H, W). Optionally it adds calibration errors (band drift, gain/offset/noise) and/or maps the
B bands back onto the 204-band grid so that a model trained on full hyperspectral data can be
fed camera data ("naive transfer", the usual workaround and our E1 baseline).
"""
from __future__ import annotations

import zlib
from typing import Optional

import numpy as np

from .cameras import CameraProfile
from .simulate import apply_radiometric, perturb_camera, srf_matrix


def back_to_full_matrix(camera: CameraProfile, wavelengths: np.ndarray) -> np.ndarray:
    """(C, B) linear interpolation from the camera's band centres back to the full grid.

    Outside the outermost band centres the nearest band's value is held constant.
    """
    wl = np.asarray(wavelengths, float)
    c = np.asarray(camera.centers, float)
    order = np.argsort(c)
    B = len(c)
    W = np.zeros((wl.size, B))
    for i, lam in enumerate(wl):
        if B == 1 or lam <= c[order[0]]:
            W[i, order[0]] = 1.0
        elif lam >= c[order[-1]]:
            W[i, order[-1]] = 1.0
        else:
            k = np.searchsorted(c[order], lam) - 1
            lo, hi = order[k], order[k + 1]
            t = (lam - c[lo]) / (c[hi] - c[lo])
            W[i, lo], W[i, hi] = 1.0 - t, t
    return W


def _seed_from(cube: np.ndarray, base: int) -> int:
    """Deterministic per-image seed (same image -> same noise, whatever the worker order)."""
    return (zlib.crc32(np.ascontiguousarray(cube[:, ::8, ::8]).tobytes()) + base) % (2 ** 32)


class SimulatedCamera:
    """Callable cube (C,H,W) -> camera image (B,H,W) or (C,H,W) if `to_full`.

    drift_seed: if given, the camera's bands are shifted/rescaled once (one physical unit).
    radiometric: add per-band gain/offset and sensor noise (fresh per image, reproducible).
    """

    def __init__(self, camera: CameraProfile, wavelengths: np.ndarray, to_full: bool = False,
                 drift_seed: Optional[int] = None, max_shift_nm: float = 10.0, fwhm_scale: float = 0.25,
                 radiometric: bool = False, noise_seed: int = 0, radiometric_kw: Optional[dict] = None):
        self.nominal = camera
        self.camera = camera if drift_seed is None else perturb_camera(
            camera, np.random.default_rng(drift_seed), max_shift_nm, fwhm_scale)
        self.M = srf_matrix(wavelengths, self.camera).astype(np.float32)            # (B, C)
        # mapping back always uses the NOMINAL centres: the user does not know the drift
        self.W = back_to_full_matrix(camera, wavelengths).astype(np.float32) if to_full else None
        self.radiometric = radiometric
        self.noise_seed = noise_seed
        self.radiometric_kw = radiometric_kw or {}
        self.n_out = len(wavelengths) if to_full else len(camera)

    def __call__(self, cube: np.ndarray) -> np.ndarray:
        mask = (cube.sum(0) > 0)[None]
        y = np.einsum("bc,chw->bhw", self.M, cube, optimize=True)
        if self.radiometric:
            rng = np.random.default_rng(_seed_from(cube, self.noise_seed))
            y = apply_radiometric(y, rng, axis=0, **self.radiometric_kw)
        if self.W is not None:
            y = np.einsum("cb,bhw->chw", self.W, y, optimize=True)
        y = np.clip(y, 1e-6, None) * mask      # keep leaf pixels positive, background exactly 0
        return y.astype(np.float32)


class Compose:
    def __init__(self, *fs):
        self.fs = fs

    def __call__(self, x):
        for f in self.fs:
            x = f(x)
        return x
