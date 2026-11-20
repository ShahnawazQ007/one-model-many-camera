"""Compare methods on the 7 held-out real camera profiles.

python scripts/plot_step7.py --baseline results/step06_camera_test/camera_eval.csv \
    --runs "SVCT (ours)=runs/svct_main" "random cameras + interp=runs/svct_interp" --out runs/step7_compare
Prints a summary table (mean over the 7 cameras per condition) and saves step7_compare.png / .csv
"""
import argparse
import pathlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--baseline", required=True, help="Step 6 camera_eval.csv (204-band model + camera-specific models)")
p.add_argument("--runs", nargs="+", default=[], help='"label=run_dir" pairs')
p.add_argument("--oracles", default=None, help='glob of extra camera-specific runs, e.g. "runs/oracle_*"')
p.add_argument("--out", default="runs/step7_compare")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)

base = pd.read_csv(a.baseline)
frames = [base[base.condition != "trained_on_this_camera"].assign(method="204-band model (Step 6)"),
          base[base.condition == "trained_on_this_camera"].assign(method="one model per camera", condition="nominal")]
if a.oracles:
    import glob, json
    extra = []
    for d in sorted(glob.glob(a.oracles)):
        f = pathlib.Path(d) / "metrics.json"
        if f.exists():
            m = json.load(open(f))
            extra.append({"camera": m["camera"], "type": "", "n_bands": 0, "condition": "nominal",
                          "cultivar_acc": m["cultivar"]["accuracy"], "cultivar_F1": m["cultivar"]["macro_f1"],
                          **{f"{t}_R2": m[t]["R2"] for t in ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer")},
                          "method": "one model per camera"})
    if extra:
        e = pd.DataFrame(extra)
        nb = base.drop_duplicates("camera").set_index("camera")["n_bands"]
        e["n_bands"] = e.camera.map(nb)
        old = frames[1]
        frames[1] = pd.concat([old[~old.camera.isin(e.camera)], e], ignore_index=True)
for spec in a.runs:
    label, d = spec.split("=", 1)
    f = pathlib.Path(d) / "camera_eval.csv"
    if f.exists():
        frames.append(pd.read_csv(f).assign(method=label))
    else:
        print("skipped (not finished):", d)
df = pd.concat(frames, ignore_index=True)
df.to_csv(out / "step7_compare.csv", index=False)

real = df[~df.camera.str.startswith("random_") & (df.camera != "hyperspectral_204")]
methods = list(dict.fromkeys(df.method))
summary = (real.groupby(["method", "condition"])[["cultivar_acc", "Fertilizer_R2", "GrainWeight_R2", "PhiPS2_R2"]]
           .mean().round(3).reset_index())
hs = df[(df.camera == "hyperspectral_204") & (df.condition == "nominal")][["method", "cultivar_acc", "Fertilizer_R2"]]
pd.set_option("display.width", 200)
print("Mean over the 7 real camera profiles (cameras the SVCT models never saw):")
print(summary.to_string(index=False))
print("\nOn the full 204-band data:")
print(hs.round(3).to_string(index=False))
rnd = df[df.camera.str.startswith("random_")]
if len(rnd):
    print("\nRandom unseen virtual cameras:")
    print(rnd.groupby("method")[["cultivar_acc", "Fertilizer_R2"]].agg(["mean", "std"]).round(3).to_string())

cams = sorted(real.camera.unique())
colors = plt.cm.tab10(np.arange(len(methods)))
fig, axes = plt.subplots(2, 1, figsize=(15, 9))
for ax, col, title in [(axes[0], "cultivar_acc", "Cultivar accuracy"), (axes[1], "Fertilizer_R2", "Fertilizer R²")]:
    w = 0.8 / len(methods)
    for k, m in enumerate(methods):
        vals = [real[(real.camera == c) & (real.method == m) & (real.condition == "nominal")][col].mean() for c in cams]
        ax.bar(np.arange(len(cams)) + (k - (len(methods) - 1) / 2) * w, vals, w, label=m, color=colors[k])
    ref = df[(df.camera == "hyperspectral_204") & (df.condition == "nominal") & (df.method == methods[0])][col]
    if len(ref):
        ax.axhline(float(ref.iloc[0]), color="k", ls="--", lw=1, label=f"204-band model on 204 bands ({float(ref.iloc[0]):.2f})")
    ax.set_xticks(np.arange(len(cams)))
    ax.set_xticklabels([f"{c}\n({int(real[real.camera == c].n_bands.iloc[0])} bands)" for c in cams], fontsize=8)
    ax.set_ylim((0, 1) if col == "cultivar_acc" else (-1, 1)); ax.axhline(0, color="grey", lw=0.6)
    ax.set_title(f"{title} on held-out camera profiles (ideal cameras)"); ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(out / "step7_compare.png", dpi=130)
print("saved", out / "step7_compare.png")
