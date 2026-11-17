"""Load one hyperspectral cube from common file formats.

Returns (cube, wavelengths): cube as float32 array (C, H, W); wavelengths in nm or None.
The band axis is found from the wavelength count when known, otherwise from typical
band counts (204 Specim IQ, 224 FX10, 249 Corning, 270-274 Headwall), otherwise the smallest axis.
"""
from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import numpy as np

TYPICAL_BANDS = (204, 224, 249, 252, 270, 274)


def _band_axis(shape, n_wl: Optional[int]) -> int:
    if n_wl:
        hits = [i for i, s in enumerate(shape) if s == n_wl]
        if hits:
            return hits[0]
    for b in TYPICAL_BANDS:
        if b in shape:
            return list(shape).index(b)
    return int(np.argmin(shape))


def load_cube(path, wavelengths=None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    p = pathlib.Path(path)
    ext = p.suffix.lower()
    wl = None
    if ext == ".npy":
        a = np.load(p)
    elif ext == ".npz":
        z = np.load(p)
        key = max(z.files, key=lambda k: z[k].size)
        a = z[key]
        for k in z.files:
            if "wave" in k.lower() or "wl" == k.lower():
                wl = np.asarray(z[k], float)
    elif ext in (".tif", ".tiff"):
        import tifffile
        a = tifffile.imread(p)
    elif ext == ".hdr":
        import spectral.io.envi as envi
        img = envi.open(str(p))
        a = np.asarray(img.load())  # (H, W, C)
        if img.metadata.get("wavelength"):
            wl = np.array(img.metadata["wavelength"], float)
    else:
        raise ValueError(f"Unsupported file type: {p.name}")
    if wavelengths is not None:
        wl = np.asarray(wavelengths, float)
    a = np.asarray(a)
    if a.ndim != 3:
        raise ValueError(f"{p.name}: expected a 3-D cube, got shape {a.shape}")
    ax = _band_axis(a.shape, len(wl) if wl is not None else None)
    cube = np.moveaxis(a, ax, 0).astype(np.float32)
    if wl is not None and len(wl) != cube.shape[0]:
        raise ValueError(f"{p.name}: {cube.shape[0]} bands but {len(wl)} wavelengths")
    return cube, wl


def find_cubes(folder, limit: int = 5):
    """First few files that look like hyperspectral cubes."""
    exts = (".hdr", ".npy", ".npz", ".tif", ".tiff")
    out = []
    for p in sorted(pathlib.Path(folder).rglob("*")):
        if p.is_file() and p.suffix.lower() in exts:
            out.append(p)
            if len(out) >= limit:
                break
    return out
