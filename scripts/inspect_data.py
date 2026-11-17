"""Inspect a downloaded dataset folder and write a report (step P0 of the plan).

Usage:
    python scripts/inspect_data.py data/raw/hyperleaf2024
    python scripts/inspect_data.py data/raw/deephs_fruit
    python scripts/inspect_data.py /kaggle/input/hyperleaf2024 reports/hyperleaf_REPORT.md

It counts files by type, opens one example of each type, prints array shapes,
value ranges and wavelengths, and looks for label tables and ID columns
(plot, fruit, leaf, cultivar ...). Output: <folder>/DATA_REPORT.md, or the
path given as a second argument (needed when the data folder is read-only, e.g. Kaggle).
"""
import collections
import pathlib
import sys

import numpy as np

ID_HINTS = ("plot", "id", "fruit", "leaf", "cultivar", "variety", "camera", "split",
            "fold", "label", "ripe", "firm", "sugar", "brix", "fert", "yield", "gsw", "psii")


def describe_array(a, name="array"):
    a = np.asarray(a)
    out = f"{name}: shape={a.shape} dtype={a.dtype}"
    if a.size and np.issubdtype(a.dtype, np.number):
        f = a.astype(float)
        out += f" min={np.nanmin(f):.4g} max={np.nanmax(f):.4g} mean={np.nanmean(f):.4g}"
    return out


def open_example(p: pathlib.Path):
    ext = p.suffix.lower()
    try:
        if ext == ".npy":
            return [describe_array(np.load(p, mmap_mode="r"), p.name)]
        if ext == ".npz":
            z = np.load(p)
            return [describe_array(z[k], f"{p.name}[{k}]") for k in z.files[:10]]
        if ext in (".tif", ".tiff"):
            import tifffile
            return [describe_array(tifffile.imread(p), p.name)]
        if ext == ".hdr":
            import spectral.io.envi as envi
            img = envi.open(str(p))
            meta = img.metadata
            wl = meta.get("wavelength")
            lines = [f"{p.name}: ENVI shape={img.shape} interleave={meta.get('interleave')}"]
            if wl:
                wl = np.array(wl, float)
                lines.append(f"  wavelengths: {len(wl)} from {wl.min():.1f} to {wl.max():.1f} nm")
            lines.append(describe_array(img.load(), "  data"))
            return lines
        if ext in (".h5", ".hdf5", ".mat"):
            try:
                import h5py
                lines = []
                with h5py.File(p, "r") as f:
                    f.visititems(lambda k, v: lines.append(f"  {k}: {getattr(v, 'shape', '')} {getattr(v, 'dtype', '')}")
                                 if hasattr(v, "shape") else None)
                return [f"{p.name}: HDF5"] + lines[:30]
            except OSError:
                from scipy.io import loadmat
                m = loadmat(p)
                return [describe_array(v, f"{p.name}[{k}]") for k, v in m.items() if not k.startswith("__")][:10]
        if ext in (".csv", ".tsv", ".xlsx", ".xls"):
            import pandas as pd
            df = pd.read_excel(p) if ext.startswith(".xls") else pd.read_csv(p, sep=None, engine="python")
            lines = [f"{p.name}: table {df.shape[0]} rows x {df.shape[1]} cols",
                     f"  columns: {list(df.columns)[:40]}"]
            for c in df.columns:
                if any(h in str(c).lower() for h in ID_HINTS):
                    nun = df[c].nunique(dropna=True)
                    miss = int(df[c].isna().sum())
                    lines.append(f"  * {c}: {nun} unique, {miss} missing, e.g. {df[c].dropna().unique()[:5].tolist()}")
            return lines
        if ext in (".png", ".jpg", ".jpeg", ".bmp"):
            return [f"{p.name}: image file ({p.stat().st_size/1e3:.0f} kB)"]
    except Exception as e:  # keep going on any unreadable file
        return [f"{p.name}: could not open ({type(e).__name__}: {e})"]
    return [f"{p.name}: (type {ext or 'none'} not inspected)"]


def main(folder, out=None):
    root = pathlib.Path(folder)
    files = [p for p in root.rglob("*") if p.is_file()]
    if not files:
        sys.exit(f"No files found in {root}")
    by_ext = collections.defaultdict(list)
    for p in files:
        by_ext[p.suffix.lower() or "(none)"].append(p)
    lines = [f"# Data report: {root.name}", "",
             f"{len(files)} files, {sum(p.stat().st_size for p in files)/1e9:.2f} GB", "",
             "| Type | Files |", "| --- | --- |"]
    lines += [f"| {e} | {len(v)} |" for e, v in sorted(by_ext.items(), key=lambda kv: -len(kv[1]))]
    lines += ["", "## Top-level folders", ""]
    lines += [f"- {d.name}/ ({sum(1 for _ in d.rglob('*') if _.is_file())} files)"
              for d in sorted(root.iterdir()) if d.is_dir()]
    lines += ["", "## Examples", ""]
    for e, v in sorted(by_ext.items()):
        tables = e in (".csv", ".tsv", ".xlsx", ".xls")
        for p in (v if tables else v[:1])[:5]:
            lines.append(f"### {p.relative_to(root)}")
            lines += ["```"] + open_example(p) + ["```", ""]
    lines += ["## Checklist (fill in)", "",
              "- [ ] Labels present for train / val / test?",
              "- [ ] Plot (or fruit) IDs present for grouped splits?",
              "- [ ] Wavelength list found? Range and number of bands:",
              "- [ ] Values are reflectance (0-1) or raw counts?",
              "- [ ] Licence / terms of use noted:"]
    out = pathlib.Path(out) if out else root / "DATA_REPORT.md"
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
    except OSError:  # read-only data folder: write next to where we run
        out = pathlib.Path("reports") / f"{root.name}_DATA_REPORT.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\nReport written to {out}")


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
