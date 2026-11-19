"""Step 7: Stochastic Virtual Camera Training (SVCT) and its ablations, then the full camera test.

Every training batch is seen through a random virtual camera (3-12 bands, random centres and widths,
optionally the full hyperspectral sensor), with random calibration error (gain/offset/noise). Two
different cameras view the same leaves and a consistency loss asks for the same answer.
The 7 real camera profiles are HELD OUT: no virtual camera may come within 15 nm of one of them.

  --stem hyper    wavelength-conditioned first layer, native bands in (our method)
  --stem interp   ordinary 204-band ResNet; virtual camera bands are interpolated back to 204 (ablation)
  --views 1 --cons 0   no consistency loss (ablation);  --radio 0   no calibration errors (ablation)

  python experiments/train_svct.py --data DATA --out runs/svct_main
Outputs: metrics.json (204-band test, same format as train_baseline), camera_eval.csv (same format as
eval_cameras.py), history.csv, config.json, spectral_stats.json, model.pt
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.cameras import list_cameras, load_camera  # noqa: E402
from anycam.data import hyperleaf, splits as S  # noqa: E402
from anycam.models import MultiTaskNet, n_params  # noqa: E402
from anycam.simulate import perturb_camera, sample_virtual_camera  # noqa: E402
from anycam.svct import (GPURenderer, SpectralStats, SVCTNet, camera_tensors, consistency_loss,  # noqa: E402
                         full_camera, geo_aug)
from anycam.train_utils import classification_metrics, r2_rmse  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--splits", default=str(ROOT / "data/splits/hyperleaf_splits.json"))
p.add_argument("--split", choices=["official", "cv"], default="official")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fold", type=int, default=0)
p.add_argument("--stem", choices=["hyper", "interp"], default="hyper")
p.add_argument("--views", type=int, choices=[1, 2], default=2)
p.add_argument("--cons", type=float, default=1.0, help="weight of the consistency loss (needs --views 2)")
p.add_argument("--radio", type=float, default=0.5, help="probability of calibration error per training sample")
p.add_argument("--p-full", type=float, default=0.15, help="probability that a batch uses the full 204 bands")
p.add_argument("--held-out", choices=["all", "none"], default="all")
p.add_argument("--backbone", default="resnet18")
p.add_argument("--epochs", type=int, default=60)
p.add_argument("--patience", type=int, default=15)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--upscale", type=float, default=2.0)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--norm-images", type=int, default=200)
p.add_argument("--drift-draws", type=int, default=3)
p.add_argument("--n-random-test", type=int, default=20)
p.add_argument("--out", default=None)
p.add_argument("--smoke", action="store_true")
a = p.parse_args()
if a.smoke:
    a.epochs, a.patience, a.batch, a.workers, a.norm_images, a.drift_draws, a.n_random_test = 2, 2, 8, 0, 8, 1, 2
if a.views == 1:
    a.cons = 0.0

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else "cpu"
out = pathlib.Path(a.out or ROOT / "runs" / f"svct_{a.stem}_v{a.views}_c{a.cons}_r{a.radio}")
out.mkdir(parents=True, exist_ok=True)
WL = hyperleaf.WAVELENGTHS
FULL = full_camera(WL)
REAL = {n: load_camera(n) for n in list_cameras()}
HELD = list(REAL.values()) if a.held_out == "all" else []

# ---------------------------------------------------------------- data (same split logic as train_baseline)
labels = hyperleaf.load_labels(a.data)
sp = pathlib.Path(a.splits)
allsp = S.load_splits(sp) if sp.exists() else {"plot_cv": S.plot_grouped_cv(labels), "official": S.official_split(labels)}
split = allsp["official"] if a.split == "official" else next(
    s for s in allsp["plot_cv"] if s["seed"] == a.seed and s["fold"] == a.fold)
tr, va, te = split["train"], split["val"], split["test"]
if a.smoke:
    tr, va, te = tr[:32], va[:16], te[:16]
print(f"device {dev} | train {len(tr)} val {len(va)} test {len(te)} leaves | stem {a.stem} views {a.views} "
      f"cons {a.cons} radio {a.radio} held-out {a.held_out}", flush=True)

images = hyperleaf.HyperLeafImages(a.data)
rng = np.random.default_rng(a.seed)
stats = SpectralStats.fit(images(labels.ImageId[i]) for i in rng.choice(tr, min(a.norm_images, len(tr)), replace=False))
json.dump(stats.state(), open(out / "spectral_stats.json", "w"))
tstats = hyperleaf.target_stats(labels, tr)
render = GPURenderer(stats, WL, dev)
TO_FULL = a.stem == "interp"


class GeoAug:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def __call__(self, c): return geo_aug(c, self.rng)


def loader(rows, train):
    ds = hyperleaf.HyperLeafDataset(labels, images, rows, transform=GeoAug(a.seed) if train else None, target_stats=tstats)
    return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=train, num_workers=a.workers,
                                       pin_memory=dev == "cuda", drop_last=train and len(rows) > a.batch,
                                       persistent_workers=a.workers > 0)


dl_tr, dl_va, dl_te = loader(tr, True), loader(va, False), loader(te, False)

# ---------------------------------------------------------------- model
if a.stem == "hyper":
    model = SVCTNet(backbone=a.backbone, pretrained=not a.smoke, upscale=a.upscale).to(dev)
else:
    model = MultiTaskNet(204, backbone=a.backbone, pretrained=not a.smoke, upscale=a.upscale).to(dev)
print(f"model {a.backbone} ({a.stem} stem): {n_params(model) / 1e6:.2f}M parameters", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


def forward(x, camera):
    with torch.autocast(dev, enabled=use_amp):
        if a.stem == "hyper":
            return model(x, *camera_tensors(camera, dev))
        return model(x)


def view(xraw, camera, nominal=None, radio=0.0, gen=None):
    return render.render(xraw, camera, nominal=nominal, to_full=TO_FULL, radiometric=radio, gen=gen)


def task_loss(o, y, t):
    lc, lr_ = o
    return F.cross_entropy(lc.float(), y) + F.mse_loss(lr_.float(), t)


cam_rng = np.random.default_rng(a.seed + 1)


def draw_camera(r):
    if r.random() < a.p_full:
        return FULL
    return sample_virtual_camera(r, held_out=HELD, fwhm_range=(10.0, 100.0), margin_scale=0.6)


vr = np.random.default_rng(12345)
VAL_CAMS = [FULL] + [sample_virtual_camera(vr, held_out=HELD, fwhm_range=(10.0, 100.0), margin_scale=0.6) for _ in range(5)]

# ---------------------------------------------------------------- training
best, best_ep, hist, t0 = 1e9, -1, [], time.time()
for ep in range(a.epochs):
    model.train()
    tot, n = 0.0, 0
    for b in dl_tr:
        x = b["cube"].to(dev, non_blocking=True); y = b["cultivar"].to(dev); t = b["targets"].to(dev)
        c1 = draw_camera(cam_rng)
        o1 = forward(view(x, c1, radio=a.radio), c1)
        loss = task_loss(o1, y, t)
        if a.views == 2:
            c2 = draw_camera(cam_rng)
            o2 = forward(view(x, c2, radio=a.radio), c2)
            loss = 0.5 * (loss + task_loss(o2, y, t)) + a.cons * consistency_loss(o1, o2)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
    model.eval()
    vtot, vn = 0.0, 0
    with torch.no_grad():
        for b in dl_va:
            x = b["cube"].to(dev); y = b["cultivar"].to(dev); t = b["targets"].to(dev)
            for c in VAL_CAMS:
                vtot += task_loss(forward(view(x, c), c), y, t).item() * len(y); vn += len(y)
    sched.step()
    vl = vtot / vn
    hist.append({"epoch": ep, "train_loss": tot / n, "val_loss": vl, "sec": round(time.time() - t0)})
    print(f"epoch {ep:2d} train {tot / n:.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if vl < best:
        best, best_ep = vl, ep
        torch.save(model.state_dict(), out / "model.pt")
    elif ep - best_ep >= a.patience:
        print("early stop", flush=True); break
pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

# ---------------------------------------------------------------- test: every camera and condition in one pass
model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
model.eval()
conds = [("hyperspectral_204", "hyperspectral", 204, "nominal", FULL, None, 0.0),
         ("hyperspectral_204", "hyperspectral", 204, "radiometric", FULL, None, 1.0)]
for name, cam in REAL.items():
    conds.append((name, cam.type, len(cam), "nominal", cam, None, 0.0))
    for k in range(1, a.drift_draws + 1):
        conds.append((name, cam.type, len(cam), f"drift{k}", perturb_camera(cam, np.random.default_rng(k)), cam, 0.0))
    conds.append((name, cam.type, len(cam), "radiometric", cam, None, 1.0))
    conds.append((name, cam.type, len(cam), "drift+radio", perturb_camera(cam, np.random.default_rng(1)), cam, 1.0))
rr = np.random.default_rng(999)
for k in range(a.n_random_test):
    vc = sample_virtual_camera(rr, fwhm_range=(10.0, 100.0), margin_scale=0.6)
    conds.append((f"random_{k:02d}", "virtual", len(vc), "nominal", vc, None, 0.0))

preds = {i: ([], []) for i in range(len(conds))}
Y, T, IDS = [], [], []
gen = torch.Generator(device=dev)
with torch.no_grad():
    for bi, b in enumerate(dl_te):
        x = b["cube"].to(dev)
        Y.append(b["cultivar"]); T.append(b["targets"]); IDS.append(b["image_id"])
        for i, (_, _, _, _, cam, nominal, radio) in enumerate(conds):
            gen.manual_seed(1000 * i + bi)
            lc, lr_ = forward(view(x, cam, nominal=nominal, radio=radio, gen=gen), nominal or cam)
            preds[i][0].append(lc.float().argmax(1).cpu()); preds[i][1].append(lr_.float().cpu())
Y = torch.cat(Y).numpy(); mean_t, std_t = tstats
T = torch.cat(T).numpy() * std_t + mean_t
rows = []
for i, (name, ctype, nb, cond, *_rest) in enumerate(conds):
    P = torch.cat(preds[i][0]).numpy(); R = torch.cat(preds[i][1]).numpy() * std_t + mean_t
    m = classification_metrics(Y, P, 4)
    r = {"camera": name, "type": ctype, "n_bands": nb, "condition": cond,
         "cultivar_acc": m["accuracy"], "cultivar_F1": m["macro_f1"]}
    for j, tname in enumerate(hyperleaf.REGRESSION_TARGETS):
        r[f"{tname}_R2"] = r2_rmse(T[:, j], R[:, j])["R2"]
    rows.append(r)
    if i == 0:
        metrics = {"split": a.split, "seed": a.seed, "fold": a.fold, "backbone": a.backbone, "best_epoch": best_ep,
                   "upscale": a.upscale, "aug": f"svct_{a.stem}_v{a.views}_c{a.cons}_r{a.radio}",
                   "camera": "hyperspectral_204", "n_test": len(Y), "cultivar": m}
        for j, tname in enumerate(hyperleaf.REGRESSION_TARGETS):
            metrics[tname] = r2_rmse(T[:, j], R[:, j])
df = pd.DataFrame(rows)
# average the drift units into one "drift" row per camera (same format as eval_cameras.py)
dr = df[df.condition.str.match(r"drift\d")]
if len(dr):
    avg = dr.groupby(["camera", "type", "n_bands"], as_index=False).mean(numeric_only=True)
    avg["condition"] = "drift"
    df = pd.concat([df[~df.condition.str.match(r"drift\d")], avg[df.columns]], ignore_index=True)
df.to_csv(out / "camera_eval.csv", index=False)
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
json.dump(vars(a), open(out / "config.json", "w"), indent=1, default=str)

pd.set_option("display.width", 220)
real = df[~df.camera.str.startswith("random_")]
order = {"nominal": 0, "drift": 1, "radiometric": 2, "drift+radio": 3}
real = real.assign(o=real.condition.map(order)).sort_values(["camera", "o"]).drop(columns="o")
print("\n" + real.round(3).to_string(index=False))
rnd = df[df.camera.str.startswith("random_")]
if len(rnd):
    print(f"\n{len(rnd)} random unseen virtual cameras: cultivar {rnd.cultivar_acc.mean():.3f} +/- {rnd.cultivar_acc.std():.3f}"
          f" | fertilizer R2 {rnd.Fertilizer_R2.mean():.3f} +/- {rnd.Fertilizer_R2.std():.3f}")
print("saved to", out, flush=True)
