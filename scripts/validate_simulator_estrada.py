"""Step 4a: does the camera simulator match a REAL MicaSense camera?

Simulates the 5 MicaSense RedEdge-MX bands from each leaf's lab spectrum (ASD, 350-2500 nm)
and compares them with the real MicaSense images of the same leaf and drying stage.
The real images are uncalibrated camera counts, so each band is compared after a linear
gain/offset fit (counts = a * reflectance + b), evaluated with leaf-grouped 5-fold CV.
"""
import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam import load_camera, simulate
from anycam.data import estrada
from anycam.data.group_splits import grouped_cv

OUT = pathlib.Path("results/step04_simulator_check")
tab, spectra = estrada.load("data/raw/estrada_leaves/DATASET")
real = pd.read_csv(OUT / "estrada_micasense_means.csv")
cam = load_camera("micasense_rededge_mx")
sim = simulate(spectra, estrada.WAVELENGTHS, cam)            # (n, 5) reflectance
sim_df = pd.DataFrame(sim, columns=[f"sim_{b.name}" for b in cam.bands])
df = pd.concat([tab.drop(columns="micasense_files").reset_index(drop=True), sim_df], axis=1)
df = df.merge(real, on=["crop", "leaf", "stage"])
real_cols = [f"b{b}_mean" for b in range(1, 6)]
sim_cols = list(sim_df.columns)

# 1) which real file band matches which simulated band? (correlation across all samples)
corr = pd.DataFrame([[df[r].corr(df[s]) for s in sim_cols] for r in real_cols], index=real_cols, columns=sim_cols)
print("Correlation (rows = real MicaSense file band, cols = simulated band):\n", corr.round(3))
from itertools import permutations
best = max(permutations(range(5)), key=lambda p: sum(corr.iloc[i, p[i]] for i in range(5)))
mapping = {real_cols[i]: sim_cols[best[i]] for i in range(5)}
print("Best one-to-one match:", mapping)

# 2) per band: cross-validated R2 of real ~ a*sim + b (leaf-grouped, so no leaf in both fit and test)
def cv_r2(sub, rc, sc):
    pred = np.full(len(sub), np.nan)
    for s in grouped_cv(sub, "leaf_id", "crop", n_folds=5, seeds=(0,)):
        tr, te = s["train"] + s["val"], s["test"]
        x, y = sub.loc[tr, sc].values, sub.loc[tr, rc].values
        a, b = np.polyfit(x, y, 1)
        pred[[sub.index.get_loc(i) for i in te]] = a * sub.loc[te, sc].values + b
    y = sub[rc].values
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)

rows = []
for scope, sub in [("all crops", df)] + [(c, df[df.crop == c]) for c in estrada.CROPS]:
    sub = sub.reset_index(drop=True)
    for rc, sc in mapping.items():
        rows.append({"scope": scope, "real_band": rc, "simulated_band": sc.replace("sim_", ""),
                     "pearson_r": round(sub[rc].corr(sub[sc]), 3), "cv_R2": round(cv_r2(sub, rc, sc), 3), "n": len(sub)})
res = pd.DataFrame(rows)
print("\n", res.to_string(index=False))
res.to_csv(OUT / "estrada_simulated_vs_real.csv", index=False)
corr.round(4).to_csv(OUT / "estrada_band_correlation_matrix.csv")
json.dump({"mapping": mapping}, open(OUT / "estrada_band_mapping.json", "w"), indent=1)

# 3) figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 5, figsize=(22, 4.5))
colors = {"Avocado": "tab:green", "Olive": "tab:olive", "Vineyard": "tab:purple"}
for a, (rc, sc) in zip(ax, mapping.items()):
    for c, g in df.groupby("crop"):
        a.scatter(g[sc], g[rc], s=5, alpha=0.4, color=colors[c], label=c)
    r = res[(res.scope == "all crops") & (res.real_band == rc)].iloc[0]
    a.set_title(f"{sc.replace('sim_', '')}: r = {r.pearson_r:.2f}, CV R² = {r.cv_R2:.2f}")
    a.set_xlabel("Simulated from lab spectrum (reflectance)"); a.set_ylabel(f"Real MicaSense {rc[:2]} (counts)")
ax[0].legend(markerscale=3)
fig.suptitle("Step 4a: simulated vs real MicaSense, 1,594 leaf samples (avocado, olive, grape; 5 drying stages)")
fig.tight_layout(); fig.savefig(OUT / "estrada_simulated_vs_real.png", dpi=110)
print("saved", OUT)
