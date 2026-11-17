"""Camera registry: one YAML file per camera in anycam/cameras/.

A camera is a list of bands, each with a centre wavelength and a FWHM (nm).
Adding a camera = adding a YAML file; no model retraining is needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

REGISTRY_DIR = Path(__file__).parent / "cameras"


@dataclass(frozen=True)
class Band:
    name: str
    center: float  # nm
    fwhm: float    # nm


@dataclass
class CameraProfile:
    name: str
    bands: List[Band]
    display_name: str = ""
    type: str = "unknown"
    source: str = ""
    note: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def centers(self):
        return [b.center for b in self.bands]

    @property
    def fwhms(self):
        return [b.fwhm for b in self.bands]

    def __len__(self):
        return len(self.bands)

    @classmethod
    def from_lists(cls, name: str, centers, fwhms, band_names: Optional[list] = None, **kw):
        if len(centers) != len(fwhms):
            raise ValueError("centers and fwhms must have the same length")
        band_names = band_names or [f"b{i}" for i in range(len(centers))]
        bands = [Band(n, float(c), float(w)) for n, c, w in zip(band_names, centers, fwhms)]
        return cls(name=name, bands=bands, **kw)


def load_camera(name_or_path: str) -> CameraProfile:
    """Load a camera by registry name (e.g. 'micasense_rededge_mx') or YAML path."""
    p = Path(name_or_path)
    if not p.suffix:
        p = REGISTRY_DIR / f"{name_or_path}.yaml"
    if not p.exists():
        raise FileNotFoundError(f"Camera '{name_or_path}' not found. Available: {list_cameras()}")
    d = yaml.safe_load(p.read_text())
    bands = [Band(b["name"], float(b["center"]), float(b["fwhm"])) for b in d["bands"]]
    for b in bands:
        if b.fwhm <= 0:
            raise ValueError(f"{p.name}: band {b.name} has non-positive FWHM")
    known = {"name", "bands", "display_name", "type", "source", "note"}
    return CameraProfile(
        name=d["name"], bands=bands, display_name=d.get("display_name", ""),
        type=d.get("type", "unknown"), source=d.get("source", ""), note=d.get("note", ""),
        meta={k: v for k, v in d.items() if k not in known},
    )


def list_cameras() -> List[str]:
    return sorted(p.stem for p in REGISTRY_DIR.glob("*.yaml"))
