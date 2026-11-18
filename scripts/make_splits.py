"""Step 3: read the three datasets, check them, and save leakage-free splits to data/splits/.

Run from the project folder:  python scripts/make_splits.py
Default locations (change with the options below):
  HyperLeaf zip:   data/data/HyperLeaf2024.zip
  DeepHS zips:     data/data/  (Kaki.zip, Mango.zip, Papaya.zip)
  DeepHS labels:   data/raw/deephs_fruit/labels/   (python scripts/get_deephs_labels.py)
  Estrada folder:  data/raw/estrada_leaves/DATASET/
"""
import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.data import deephs, estrada, hyperleaf, splits  # noqa: E402
from anycam.data.group_splits import grouped_cv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--hyperleaf", default=ROOT / "data/data/HyperLeaf2024.zip")
ap.add_argument("--deephs-zips", default=ROOT / "data/data")
ap.add_argument("--deephs-labels", default=ROOT / "data/raw/deephs_fruit/labels")
ap.add_argument("--estrada", default=ROOT / "data/raw/estrada_leaves/DATASET")
ap.add_argument("--out", default=ROOT / "data/splits")
ap.add_argument("--check-images", type=int, default=3, help="read this many images per dataset as a check")
a = ap.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
summary = {}

# ---------------------------------------------------------------- HyperLeaf
print("== HyperLeaf2024")
df = hyperleaf.load_labels(a.hyperleaf)
info = hyperleaf.check_plots(df)
print("  ", info)
df.drop(columns=[c for c in df.columns if c in hyperleaf.CULTIVARS]).to_csv(out / "hyperleaf_labels.csv", index=False)
cv = splits.plot_grouped_cv(df)
for s in cv:
    splits.assert_no_leakage(df, s)
off = splits.official_split(df)
splits.assert_no_leakage(df, off)
splits.save_splits({"plot_cv": cv, "official": off}, out / "hyperleaf_splits.json")
imgs = hyperleaf.HyperLeafImages(a.hyperleaf)
for i in df.ImageId[: a.check_images]:
    c = imgs(i)
    print(f"   image {i}: {c.shape}, reflectance {c.min():.3f}..{c.max():.3f}, mean {c.mean():.3f}")
summary["hyperleaf"] = {**info, "cv_splits": len(cv),
                        "test_leaves_per_fold": [len(s["test"]) for s in cv if s["seed"] == 0]}

# ---------------------------------------------------------------- DeepHS fruit
print("== DeepHS fruit")
labels_dir = pathlib.Path(a.deephs_labels)
if (labels_dir / "test_v2.json").exists():
    fr, wl = deephs.load_labels(labels_dir)
    fr.to_csv(out / "deephs_labels.csv", index=False)
    json.dump({k: v.tolist() for k, v in wl.items()}, open(out / "deephs_wavelengths.json", "w"))
    cv = grouped_cv(fr, "fruit_id", "ripeness")
    for s in cv:
        splits.assert_no_leakage(fr, s, group_col="fruit_id")
    splits.save_splits({"fruit_cv": cv}, out / "deephs_splits.json")
    pairs = deephs.camera_pairs(fr)
    reader = deephs.DeepHSImages(a.deephs_zips)
    for _, r in fr.head(a.check_images).iterrows():
        c = reader(r)
        print(f"   {r.fruit} {r.camera} {r.side}: {c.shape}, values {c.min():.3f}..{c.max():.3f}")
    summary["deephs"] = {"recordings": len(fr), "fruits": int(fr.fruit_id.nunique()),
                         "fruits_per_kind": fr.groupby("fruit").fruit_id.nunique().to_dict(),
                         "camera_pairs": len(pairs), "ripeness": fr.drop_duplicates("fruit_id").ripeness.value_counts().to_dict(),
                         "with_sugar": int(fr.sugar.notna().sum())}
    print("  ", summary["deephs"])
else:
    print("   labels missing: run  python scripts/get_deephs_labels.py  (or copy the 'dataset' folder, see docs)")

# ---------------------------------------------------------------- Estrada leaves
print("== Estrada leaves")
if pathlib.Path(a.estrada).exists():
    tab, spectra = estrada.load(a.estrada)
    tab.drop(columns="micasense_files").to_csv(out / "estrada_labels.csv", index=False)
    cv = grouped_cv(tab, "leaf_id", "crop")
    for s in cv:
        splits.assert_no_leakage(tab, s, group_col="leaf_id")
    splits.save_splits({"leaf_cv": cv}, out / "estrada_splits.json")
    summary["estrada"] = {"samples": len(tab), "leaves": int(tab.leaf_id.nunique()),
                          "leaves_per_crop": tab.drop_duplicates("leaf_id").crop.value_counts().to_dict(),
                          "micasense_ok": int(tab.micasense_ok.sum()),
                          "spectra_range": [float(spectra.min()), float(spectra.max())]}
    print("  ", summary["estrada"])
else:
    print("   folder not found:", a.estrada)

(out / "SUMMARY.json").write_text(json.dumps(summary, indent=1, default=str))
print("\nSaved splits and summary to", out)
