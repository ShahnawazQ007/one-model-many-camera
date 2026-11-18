"""Estrada et al. 2025 (Sci. Reports): avocado, olive and grape (vineyard) leaves,
each at 5 drying stages, measured with an ASD spectrometer (350-2500 nm, 1 nm) and
a real MicaSense camera (5 band images per leaf and stage).

Folder: data/raw/estrada_leaves/DATASET/<Crop>/  with Spectral_<crop>.mat, label .mat files
and 'Multispectral Images/leafNNNdS_B.tif' (S = stage 0-4, B = band 1-5).
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

CROPS = {"Avocado": "avocado", "Olive": "olive", "Vineyard": "vineyard"}
STAGES = ["fresh", "stage1", "stage2", "stage3", "dry"]       # stage index 0..4 = day 0..4
WAVELENGTHS = np.arange(350, 2501, dtype=float)               # 2151 values
LABELS = {"chlorophyll": "Chlorophyll", "nitrogen": "Nitrogen", "FMC_d": "FMC", "FMC_f": "FMC", "weight": "Weight"}
DAMAGED = {("Olive", 4, 2)}  # (crop, leaf, stage) with a damaged MicaSense file in the download


def _mat(path):
    from scipy.io import loadmat
    return {k: np.asarray(v) for k, v in loadmat(path).items() if not k.startswith("__")}


def _find(folder: pathlib.Path, prefix: str) -> pathlib.Path:
    hits = [p for p in folder.glob("*.mat") if p.name.lower().startswith(prefix.lower())]
    if not hits:
        raise FileNotFoundError(f"{prefix}*.mat not found in {folder}")
    return hits[0]


def load(root) -> tuple[pd.DataFrame, np.ndarray]:
    """Returns (table, spectra): one table row per (crop, leaf, stage); spectra[i] is that row's 2151-value spectrum."""
    root = pathlib.Path(root)
    rows, spectra = [], []
    for crop, low in CROPS.items():
        folder = root / crop
        spec = _mat(_find(folder, "Spectral"))
        allm = {}
        for prefix in ("Chlorophyll", "Nitrogen", "FMC", "Weight"):
            allm.update({k.lower(): v for k, v in _mat(_find(folder, prefix)).items()})
        labels = {}
        for key in LABELS:
            hit = [k for k in allm if k.startswith(key.lower() + "_")]
            if not hit:
                raise KeyError(f"{crop}: no '{key}' array in the label files ({list(allm)})")
            labels[key] = allm[hit[0]]
        n_leaves = labels["chlorophyll"].shape[0]
        for s, stage in enumerate(STAGES):
            arr = spec[[k for k in spec if k.lower().startswith(stage)][0]]
            for leaf in range(n_leaves):
                tifs = [folder / "Multispectral Images" / f"leaf{leaf:03d}d{s}_{b}.tif" for b in range(1, 6)]
                rows.append({"crop": crop, "leaf": leaf, "stage": s, "leaf_id": f"{crop}_{leaf:03d}",
                             **{k: float(v[leaf, s]) for k, v in labels.items()},
                             "micasense_ok": all(p.exists() for p in tifs) and (crop, leaf, s) not in DAMAGED,
                             "micasense_files": [str(p) for p in tifs]})
                spectra.append(arr[leaf])
    return pd.DataFrame(rows), np.asarray(spectra, dtype=np.float32)


def read_micasense(row) -> np.ndarray:
    """(5, H, W) raw MicaSense values for one leaf and stage (needs the imagecodecs package)."""
    import tifffile
    return np.stack([tifffile.imread(p) for p in row["micasense_files"]]).astype(np.float32)
