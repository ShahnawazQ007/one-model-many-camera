"""Step 4b: do two REAL hyperspectral cameras agree after camera simulation?

For each labelled fruit side, simulate the same low-cost cameras from Specim FX10 (VIS)
and from Corning microHSI 410 (VIS_COR). Both record reflectance, so no fitting is used.
If the simulator and the cameras are consistent, the two simulated readings should agree.
"""
import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam import load_camera, simulate, CameraProfile

OUT = pathlib.Path("results/step04_simulator_check")
wl = {k: np.array(v) for k, v in json.load(open("data/splits/deephs_wavelengths.json")).items()}
d = pd.read_csv(OUT / "deephs_mean_spectra.csv")
d["spec"] = d.spectrum.map(lambda s: np.array(s.split(), float))
a = d[d.camera == "VIS"].set_index(["fruit_id", "side"]); b = d[d.camera == "VIS_COR"].set_index(["fruit_id", "side"])
pairs = a.join(b, lsuffix="_fx10", rsuffix="_cor", how="inner").reset_index()
print("fruit sides with both cameras:", len(pairs))

# 1) whole-spectrum agreement on the shared range (410-900 nm)
grid = np.arange(410, 901, 2.0)
fx = np.stack([np.interp(grid, wl["VIS"], s) for s in pairs.spec_fx10])
co = np.stack([np.interp(grid, wl["VIS_COR"], s) for s in pairs.spec_cor])
shape_r = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(fx, co)])
print(f"spectral shape correlation per fruit side: median {np.median(shape_r):.3f} (IQR {np.percentile(shape_r,25):.3f}-{np.percentile(shape_r,75):.3f})")
print(f"mean absolute reflectance difference: {np.mean(np.abs(fx - co)):.4f}")

# 2) simulated low-cost cameras from each real camera
cams = ["micasense_rededge_mx", "parrot_sequoia", "dji_p4_multispectral", "generic_rgb"]
s2 = load_camera("sentinel2a_vnir")
s2 = CameraProfile(name="sentinel2a_408_901", bands=[x for x in s2.bands if 430 < x.center < 880])
rows, store = [], {}
for name in cams + ["sentinel2a_408_901"]:
    cam = s2 if name.startswith("sentinel") else load_camera(name)
    x = simulate(np.stack(pairs.spec_fx10), wl["VIS"], cam)
    y = simulate(np.stack(pairs.spec_cor), wl["VIS_COR"], cam)
    store[name] = (x, y, [bb.name for bb in cam.bands])
    for j, bb in enumerate(cam.bands):
        xi, yi = x[:, j], y[:, j]
        for scope, m in [("all", np.ones(len(pairs), bool))] + [(f, (pairs.fruit_fx10 == f).values) for f in ("Kaki", "Mango", "Papaya")]:
            rows.append({"camera": name, "band": bb.name, "center_nm": bb.center, "scope": scope, "n": int(m.sum()),
                         "pearson_r": round(float(np.corrcoef(xi[m], yi[m])[0, 1]), 3),
                         "R2_no_fit": round(float(1 - np.sum((yi[m] - xi[m]) ** 2) / np.sum((yi[m] - yi[m].mean()) ** 2)), 3),
                         "mean_abs_diff": round(float(np.mean(np.abs(xi[m] - yi[m]))), 4),
                         "mean_refl_fx10": round(float(xi[m].mean()), 4), "mean_refl_corning": round(float(yi[m].mean()), 4)})
res = pd.DataFrame(rows); res.to_csv(OUT / "deephs_fx10_vs_corning_simulated.csv", index=False)
print(res[res.scope == "all"].drop(columns="scope").to_string(index=False))
print("\nper fruit (MicaSense):")
print(res[(res.camera == "micasense_rededge_mx") & (res.scope != "all")][["scope", "band", "pearson_r", "R2_no_fit", "mean_abs_diff"]].to_string(index=False))

# 3) figure: MicaSense bands, FX10-simulated vs Corning-simulated
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
x, y, names = store["micasense_rededge_mx"]
fig, ax = plt.subplots(1, 6, figsize=(26, 4.4))
col = {"Kaki": "tab:orange", "Mango": "gold", "Papaya": "tab:red"}
for j, nm in enumerate(names):
    for f in col:
        m = (pairs.fruit_fx10 == f).values
        ax[j].scatter(x[m, j], y[m, j], s=10, alpha=0.6, color=col[f], label=f)
    lo, hi = min(x[:, j].min(), y[:, j].min()), max(x[:, j].max(), y[:, j].max())
    ax[j].plot([lo, hi], [lo, hi], "k--", lw=1)
    r = res[(res.camera == "micasense_rededge_mx") & (res.band == nm) & (res.scope == "all")].iloc[0]
    ax[j].set_title(f"{nm}: r={r.pearson_r:.2f}, R²={r.R2_no_fit:.2f}")
    ax[j].set_xlabel("simulated from Specim FX10"); ax[j].set_ylabel("simulated from Corning")
ax[0].legend()
k = np.argsort(shape_r)[len(shape_r) // 2]
ax[5].plot(grid, fx[k], label="Specim FX10"); ax[5].plot(grid, co[k], label="Corning")
ax[5].set_title(f"median example: {pairs.fruit_id[k]} {pairs.side[k]}"); ax[5].set_xlabel("nm"); ax[5].legend()
fig.suptitle("Step 4b: the same MicaSense camera simulated from two real hyperspectral cameras, same fruit side (no fitting)")
fig.tight_layout(); fig.savefig(OUT / "deephs_fx10_vs_corning_simulated.png", dpi=110)
print("saved")
