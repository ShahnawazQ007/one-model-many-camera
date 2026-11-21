"""Summarise the Step 9 cross-validation (runs/cv): mean over the 7 held-out camera profiles per fold,
then mean (std) over folds, plus how many folds each SVCT model beats the baseline.

python scripts/summarize_cv.py --cv runs/cv --out runs/cv_summary
"""
import argparse
import glob
import pathlib
import re

import pandas as pd

p = argparse.ArgumentParser()
p.add_argument("--cv", nargs="+", default=["runs/cv"], help="one or more cv folders (e.g. from two notebooks)")
p.add_argument("--out", default="runs/cv_summary")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
LABEL = {"baseline_eval": "204-band baseline", "svct": "SVCT (wavelength stem)", "interp": "SVCT (interp)"}
frames = []
for root in a.cv:
    for f in glob.glob(f"{root}/*/camera_eval.csv"):
        m = re.match(r"(baseline_eval|svct|interp)_s(\d+)_f(\d+)$", pathlib.Path(f).parent.name)
        if m:
            frames.append(pd.read_csv(f).assign(method=LABEL[m[1]], seed=int(m[2]), fold=int(m[3])))
df = pd.concat(frames, ignore_index=True)
df.to_csv(out / "cv_all_cameras.csv", index=False)
COLS = ["cultivar_acc", "Fertilizer_R2", "GrainWeight_R2", "PhiPS2_R2"]
real = df[~df.camera.str.startswith("random_") & (df.camera != "hyperspectral_204")]
per_fold = real.groupby(["method", "condition", "seed", "fold"])[COLS].mean().reset_index()
per_fold.to_csv(out / "cv_per_fold.csv", index=False)
n = per_fold.groupby(["method", "condition"]).size().rename("n_folds")
fmt = lambda v: f"{v.mean():.3f} ({v.std():.3f})"
summ = per_fold.groupby(["method", "condition"])[COLS].agg(fmt).join(n)
pd.set_option("display.width", 220)
print("Mean (std over folds) of the mean over the 7 held-out camera profiles:")
print(summ.to_string())
summ.to_csv(out / "cv_summary.csv")
hs = df[(df.camera == "hyperspectral_204") & (df.condition == "nominal")]
print("\nOn the full 204-band data (nominal):")
print(hs.groupby("method")[COLS].agg(fmt).to_string())
rnd = df[df.camera.str.startswith("random_")]
if len(rnd):
    print("\nRandom unseen virtual cameras (SVCT runs only):")
    print(rnd.groupby(["method", "condition"])[COLS[:2]].mean().round(3).to_string())
nom = per_fold[per_fold.condition == "nominal"].set_index(["seed", "fold", "method"])[COLS].unstack("method")
print("\nPer fold, nominal, mean over 7 cameras (cultivar acc | fertilizer R2):")
print(nom[["cultivar_acc", "Fertilizer_R2"]].round(3).to_string())
for m in ("SVCT (wavelength stem)", "SVCT (interp)"):
    if ("cultivar_acc", m) in nom and ("cultivar_acc", "204-band baseline") in nom:
        for c in ("cultivar_acc", "Fertilizer_R2"):
            d = (nom[(c, m)] - nom[(c, "204-band baseline")]).dropna()
            print(f"{m} vs baseline, {c}: mean gain {d.mean():+.3f}, better in {(d > 0).sum()}/{len(d)} folds")
