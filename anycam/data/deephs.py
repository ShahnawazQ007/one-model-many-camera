"""DeepHS Fruit (kaki, mango, papaya) recorded with two cameras:
VIS = Specim FX10 (224 bands, 398-1004 nm), VIS_COR = Corning microHSI 410 (249 bands, 408-901 nm).

Labels and wavelength lists come from the DeepHS code repository
(github.com/cogsys-tuebingen/deephs_fruit, dataset/*_v2.json); images are read straight
from Kaki.zip / Mango.zip / Papaya.zip (ENVI .hdr + .bin, float32).

Each labelled fruit has 4 recordings: 2 cameras x (front, back). The repository's own
train/val/test lists put the same fruit in more than one list, so we split by FRUIT.
"""
from __future__ import annotations

import io
import json
import os
import pathlib
import zipfile
from typing import Dict, Optional

import numpy as np
import pandas as pd

FRUITS = ("Kaki", "Mango", "Papaya")
CAMERAS = ("VIS", "VIS_COR")
RIPENESS = {"unripe": 0, "perfect": 1, "overripe": 2}
LABEL_FILES = ("train_only_labeled_v2.json", "val_v2.json", "test_v2.json")
_ENVI_DTYPES = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32, 5: np.float64, 12: np.uint16}


def load_labels(label_dir) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """One row per labelled recording + the wavelength list of each camera."""
    rows, wavelengths = [], {}
    for name in LABEL_FILES:
        d = json.loads((pathlib.Path(label_dir) / name).read_text())
        for c in d["cameras"]:
            wavelengths[c["id"]] = np.asarray(c["wavelengths"], float)
        rec = {r["id"]: r for r in d["records"]}
        for a in d["annotations"]:
            r = rec[a["record_id"]]
            if r["fruit"] not in FRUITS or r["camera_type"] not in CAMERAS:
                continue
            stem = pathlib.Path(r["files"]["data_file"]).stem
            rows.append({
                "fruit": r["fruit"], "camera": r["camera_type"], "side": r["side"], "day": r["day"],
                "fruit_id": f'{r["fruit"]}_{r["day"]}_{stem.split("_")[-2]}',
                "data_file": r["files"]["data_file"], "header_file": r["files"]["header_file"],
                "ripeness": a["ripeness_state"], "ripeness_idx": RIPENESS.get(a["ripeness_state"], -1),
                "firmness": a.get("firmness"), "sugar": a.get("sugar_content"),
                "repo_list": name.replace("_v2.json", ""),
            })
    df = pd.DataFrame(rows).drop_duplicates(subset=["data_file"]).reset_index(drop=True)
    return df, wavelengths


def parse_envi_header(text: str) -> dict:
    h = {}
    for line in text.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            h[k.strip().lower()] = v.strip()
    return h


def envi_from_bytes(hdr_text: str, data: bytes) -> np.ndarray:
    """ENVI cube -> float32 (bands, lines, samples)."""
    h = parse_envi_header(hdr_text)
    lines, samples, bands = int(h["lines"]), int(h["samples"]), int(h["bands"])
    dt = np.dtype(_ENVI_DTYPES[int(h["data type"])]).newbyteorder(">" if h.get("byte order", "0") == "1" else "<")
    off = int(h.get("header offset", 0))
    a = np.frombuffer(data, dtype=dt, offset=off, count=lines * samples * bands)
    il = h.get("interleave", "bip").lower()
    if il == "bip":
        a = a.reshape(lines, samples, bands).transpose(2, 0, 1)
    elif il == "bil":
        a = a.reshape(lines, bands, samples).transpose(1, 0, 2)
    elif il == "bsq":
        a = a.reshape(bands, lines, samples)
    else:
        raise ValueError(f"unknown interleave {il}")
    return np.ascontiguousarray(a, dtype=np.float32)


class DeepHSImages:
    """Reads a recording from <zip_dir>/<Fruit>.zip, or from an unzipped <zip_dir>/<Fruit>/ folder."""

    def __init__(self, zip_dir):
        self.zip_dir = pathlib.Path(zip_dir)
        self._zips: Dict[str, zipfile.ZipFile] = {}
        self._pid: Optional[int] = None

    def _read(self, member: str) -> bytes:
        fruit = member.split("/")[0]
        folder = self.zip_dir / member
        if folder.exists():
            return folder.read_bytes()
        if self._pid != os.getpid():
            self._zips, self._pid = {}, os.getpid()
        if fruit not in self._zips:
            self._zips[fruit] = zipfile.ZipFile(self.zip_dir / f"{fruit}.zip")
        return self._zips[fruit].read(member)

    def __call__(self, row) -> np.ndarray:
        return envi_from_bytes(self._read(row["header_file"]).decode("ascii", "ignore"), self._read(row["data_file"]))


def camera_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the same fruit and side exist on both cameras: columns *_VIS and *_VIS_COR."""
    a = df[df.camera == "VIS"].set_index(["fruit_id", "side"])
    b = df[df.camera == "VIS_COR"].set_index(["fruit_id", "side"])
    return a.join(b, lsuffix="_VIS", rsuffix="_VIS_COR", how="inner").reset_index()
