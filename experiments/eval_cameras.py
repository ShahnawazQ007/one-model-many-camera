"""Step 6 (E1 + E2): test a model trained on full 204-band hyperspectral data on other cameras.

For every camera profile in anycam/cameras/ the test leaves are simulated as that camera would
record them, mapped back to the 204-band grid by linear interpolation (the usual workaround when a
model expects hyperspectral input) and fed to the trained model. Conditions:
  nominal       ideal camera, exactly as in its datasheet
  drift         the physical unit differs from its datasheet: band centres +/-10 nm, widths +/-25 %
                (averaged over --drift-draws random units)
  radiometric   per-band gain 0.8-1.25, offset +/-0.02, sensor noise SNR 20-50 dB
  drift+radio   both
Optionally, models trained directly on one camera (`train_baseline.py --camera X`) are added as
"camera-specific" reference points (--oracles).

  python experiments/eval_cameras.py --data DATA --run runs/A_resnet18_up2_strong --out runs/step6_eval \
      --oracles "runs/oracle_*"
Outputs: camera_eval.csv, camera_eval.png
"""
import argparse
import glob
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.cameras import list_cameras, load_camera  # noqa: E402
from anycam.camera_tf import Compose, SimulatedCamera  # noqa: E402
from anycam.data import hyperleaf, splits as S  # noqa: E402
from anycam.models import MultiTaskNet  # noqa: E402
from anycam.simulate import apply_radiometric  # noqa: E402
from anycam.train_utils import BandNorm, classification_metrics, r2_rmse  # noqa: E402
from anycam.camera_tf import _seed_from  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--run", required=True, help="folder of the 204-band model (model.pt, band_norm.json, config.json)")
p.add_argument("--cameras", nargs="*", default=None, help="default: all profiles in anycam/cameras")
p.add_argument("--drift-draws", type=int, default=3)
p.add_argument("--oracles", default=None, help='glob of camera-specific runs, e.g. "runs/oracle_*"')
p.add_argument("--batch", type=int, default=64)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--limit", type=int, default=0, help="only the first N test leaves (quick check)")
p.add_argument("--out", default="runs/step6_eval")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"
run = pathlib.Path(a.run)
cfg = json.load(open(run / "config.json"))
WL = hyperleaf.WAVELENGTHS

# ---------------------------------------------------------------- same split + target scaling as training
labels = hyperleaf.load_labels(a.data)
sp_path = pathlib.Path(cfg.get("splits", ""))
allsp = S.load_splits(sp_path) if sp_path.exists() else {"plot_cv": S.plot_grouped_cv(labels), "official": S.official_split(labels)}
split = allsp["official"] if cfg["split"] == "official" else next(
    s for s in allsp["plot_cv"] if s["seed"] == cfg["seed"] and s["fold"] == cfg["fold"])
tr, te = split["train"], split["test"]
if a.limit:
    te = te[:a.limit]
mean_t, std_t = hyperleaf.target_stats(labels, tr)
ns = json.load(open(run / "band_norm.json"))
norm = BandNorm(ns["mean"], ns["std"])
images = hyperleaf.HyperLeafImages(a.data)

model = MultiTaskNet(204, backbone=cfg["backbone"], pretrained=False, upscale=cfg.get("upscale", 1.0))
model.load_state_dict(torch.load(run / "model.pt", map_location="cpu"))
model.to(dev).eval()


class Radiometric204:
    """Calibration error applied directly to the 204-band data (hyperspectral reference under E2)."""
    def __call__(self, cube):
        mask = (cube.sum(0) > 0)[None]
        y = apply_radiometric(cube, np.random.default_rng(_seed_from(cube, 0)), axis=0)
        return (np.clip(y, 1e-6, None) * mask).astype(np.float32)


def evaluate(transform):
    ds = hyperleaf.HyperLeafDataset(labels, images, te, transform=Compose(transform, norm),
                                    target_stats=(mean_t, std_t))
    dl = torch.utils.data.DataLoader(ds, batch_size=a.batch, num_workers=a.workers, pin_memory=dev == "cuda")
    P, R, Y, T = [], [], [], []
    with torch.no_grad(), torch.autocast(dev, enabled=dev == "cuda"):
        for b in dl:
            lc, lr_ = model(b["cube"].to(dev, non_blocking=True))
            P.append(lc.float().argmax(1).cpu()); R.append(lr_.float().cpu()); Y.append(b["cultivar"]); T.append(b["targets"])
    P, R, Y, T = (torch.cat(v).numpy() for v in (P, R, Y, T))
    R, T = R * std_t + mean_t, T * std_t + mean_t
    m = classification_metrics(Y, P, 4)
    row = {"cultivar_acc": m["accuracy"], "cultivar_F1": m["macro_f1"]}
    for j, name in enumerate(hyperleaf.REGRESSION_TARGETS):
        row[f"{name}_R2"] = r2_rmse(T[:, j], R[:, j])["R2"]
    return row


rows, t0 = [], time.time()


def record(camera, ctype, nb, cond, transform):
    r = {"camera": camera, "type": ctype, "n_bands": nb, "condition": cond, **evaluate(transform)}
    rows.append(r)
    print(f"{camera:24s} {cond:12s} cultivar {r['cultivar_acc']:.3f}  fertilizer R2 {r['Fertilizer_R2']:+.3f}"
          f"  grain R2 {r['GrainWeight_R2']:+.3f}  ({time.time() - t0:.0f}s)", flush=True)


record("hyperspectral_204", "hyperspectral", 204, "nominal", Compose())
record("hyperspectral_204", "hyperspectral", 204, "radiometric", Radiometric204())
for name in (a.cameras or list_cameras()):
    cam = load_camera(name)
    record(name, cam.type, len(cam), "nominal", SimulatedCamera(cam, WL, to_full=True))
    drift = [evaluate(SimulatedCamera(cam, WL, to_full=True, drift_seed=k)) for k in range(1, a.drift_draws + 1)]
    r = {"camera": name, "type": cam.type, "n_bands": len(cam), "condition": "drift",
         **{k: float(np.mean([d[k] for d in drift])) for k in drift[0]}}
    rows.append(r)
    print(f"{name:24s} {'drift':12s} cultivar {r['cultivar_acc']:.3f}  fertilizer R2 {r['Fertilizer_R2']:+.3f}"
          f"  (mean of {a.drift_draws} units, {time.time() - t0:.0f}s)", flush=True)
    record(name, cam.type, len(cam), "radiometric", SimulatedCamera(cam, WL, to_full=True, radiometric=True))
    record(name, cam.type, len(cam), "drift+radio", SimulatedCamera(cam, WL, to_full=True, drift_seed=1, radiometric=True))

# ---------------------------------------------------------------- camera-specific models (if given)
if a.oracles:
    for d in sorted(glob.glob(a.oracles)):
        f = pathlib.Path(d) / "metrics.json"
        if not f.exists():
            continue
        m = json.load(open(f))
        cam = m.get("camera")
        r = {"camera": cam, "type": load_camera(cam).type, "n_bands": len(load_camera(cam)),
             "condition": "trained_on_this_camera",
             "cultivar_acc": m["cultivar"]["accuracy"], "cultivar_F1": m["cultivar"]["macro_f1"],
             **{f"{t}_R2": m[t]["R2"] for t in hyperleaf.REGRESSION_TARGETS}}
        rows.append(r)

df = pd.DataFrame(rows)
df.to_csv(out / "camera_eval.csv", index=False)
pd.set_option("display.width", 200)
print("\n" + df.round(3).to_string(index=False))

# ---------------------------------------------------------------- figure
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

conds = ["nominal", "drift", "radiometric", "drift+radio", "trained_on_this_camera"]
colors = {"nominal": "#1f77b4", "drift": "#ff7f0e", "radiometric": "#9467bd", "drift+radio": "#d62728",
          "trained_on_this_camera": "#2ca02c"}
cams = [c for c in df.camera.unique() if c != "hyperspectral_204"]
ref = df[(df.camera == "hyperspectral_204") & (df.condition == "nominal")].iloc[0]
fig, axes = plt.subplots(2, 1, figsize=(14, 9))
for ax, col, title in [(axes[0], "cultivar_acc", "Cultivar accuracy"), (axes[1], "Fertilizer_R2", "Fertilizer R²")]:
    present = [c for c in conds if c in set(df.condition)]
    w = 0.8 / len(present)
    for k, cond in enumerate(present):
        vals = [df[(df.camera == c) & (df.condition == cond)][col].mean() for c in cams]
        ax.bar(np.arange(len(cams)) + (k - (len(present) - 1) / 2) * w, vals, w, label=cond, color=colors[cond])
    ax.axhline(ref[col], color="k", ls="--", label=f"full hyperspectral (204 bands): {ref[col]:.2f}")
    if col == "cultivar_acc":
        ax.axhline(0.25, color="grey", ls=":", label="random guess (4 cultivars)")
        ax.set_ylim(0, 1)
    else:
        ax.set_ylim(max(-1.0, float(np.nanmin(df[col])) - 0.05), 1)
        ax.axhline(0, color="grey", lw=0.8)
    ax.set_xticks(np.arange(len(cams)))
    ax.set_xticklabels([f"{c}\n({int(df[df.camera == c].n_bands.iloc[0])} bands)" for c in cams], fontsize=8)
    ax.set_title(f"{title}: 204-band model tested on other cameras"); ax.legend(fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(out / "camera_eval.png", dpi=130)
print("saved", out / "camera_eval.csv", out / "camera_eval.png")
