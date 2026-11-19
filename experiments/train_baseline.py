"""Step 5: baseline = full-band (204-band) ImageNet CNN on HyperLeaf2024.

Examples
  python experiments/train_baseline.py --data data/data/HyperLeaf2024.zip --split official
  python experiments/train_baseline.py --data /kaggle/input/hyperleaf2024 --split cv --seed 0 --fold 0
  python experiments/train_baseline.py --data ... --smoke        # tiny CPU check (2 epochs, 32 leaves)

Outputs (in --out): metrics.json, predictions.csv, history.csv, config.json, band_norm.json, model.pt
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
from anycam.data import hyperleaf, splits as S  # noqa: E402
from anycam.models import MultiTaskNet, n_params  # noqa: E402
from anycam.cameras import load_camera  # noqa: E402
from anycam.camera_tf import Compose, SimulatedCamera  # noqa: E402
from anycam.train_utils import BandNorm, classification_metrics, r2_rmse, random_flip, strong_aug  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True, help="HyperLeaf2024.zip or the unzipped folder")
p.add_argument("--splits", default=str(ROOT / "data/splits/hyperleaf_splits.json"))
p.add_argument("--split", choices=["official", "cv"], default="official")
p.add_argument("--seed", type=int, default=0)
p.add_argument("--fold", type=int, default=0)
p.add_argument("--backbone", default="resnet18")
p.add_argument("--pretrained", type=int, default=1)
p.add_argument("--epochs", type=int, default=40)
p.add_argument("--patience", type=int, default=8)
p.add_argument("--batch", type=int, default=32)
p.add_argument("--lr", type=float, default=3e-4)
p.add_argument("--wd", type=float, default=0.05)
p.add_argument("--workers", type=int, default=4)
p.add_argument("--norm-images", type=int, default=200)
p.add_argument("--out", default=None)
p.add_argument("--upscale", type=float, default=1.0, help="enlarge images inside the model (e.g. 2)")
p.add_argument("--aug", choices=["flip", "strong"], default="flip")
p.add_argument("--camera", default=None, help="train on a simulated camera (e.g. micasense_rededge_mx) instead of 204 bands")
p.add_argument("--smoke", action="store_true")
a = p.parse_args()

torch.manual_seed(a.seed); np.random.seed(a.seed)
dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
if a.smoke:
    a.epochs, a.patience, a.batch, a.workers, a.norm_images, a.pretrained = 2, 2, 8, 0, 8, 0
out = pathlib.Path(a.out or ROOT / "runs" / f"baseline_{a.camera or 'hs204'}_{a.backbone}_{a.split}_s{a.seed}_f{a.fold}")
out.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- data and split
labels = hyperleaf.load_labels(a.data)
hyperleaf.check_plots(labels)
sp_path = pathlib.Path(a.splits)
if sp_path.exists():
    allsp = S.load_splits(sp_path)
else:  # compute on the fly (e.g. on Kaggle)
    allsp = {"plot_cv": S.plot_grouped_cv(labels), "official": S.official_split(labels)}
split = allsp["official"] if a.split == "official" else next(
    s for s in allsp["plot_cv"] if s["seed"] == a.seed and s["fold"] == a.fold)
S.assert_no_leakage(labels, split)
tr, va, te = split["train"], split["val"], split["test"]
if a.smoke:
    tr, va, te = tr[:32], va[:16], te[:16]
print(f"device {dev} | train {len(tr)} val {len(va)} test {len(te)} leaves", flush=True)

images = hyperleaf.HyperLeafImages(a.data)
cam = SimulatedCamera(load_camera(a.camera), hyperleaf.WAVELENGTHS) if a.camera else None
raw = cam if cam else Compose()  # Compose() with no steps = identity
IN_CH = cam.n_out if cam else 204
rng = np.random.default_rng(a.seed)
norm = BandNorm.fit(raw(images(labels.ImageId[i])) for i in rng.choice(tr, min(a.norm_images, len(tr)), replace=False))
json.dump(norm.state(), open(out / "band_norm.json", "w"))
tstats = hyperleaf.target_stats(labels, tr)


class Aug:
    def __init__(self, seed): self.rng = np.random.default_rng(seed)
    def __call__(self, c): return (strong_aug if a.aug == 'strong' else random_flip)(norm(raw(c)), self.rng)


def loader(rows, train):
    ds = hyperleaf.HyperLeafDataset(labels, images, rows, transform=Aug(a.seed) if train else Compose(raw, norm), target_stats=tstats)
    return torch.utils.data.DataLoader(ds, batch_size=a.batch, shuffle=train, num_workers=a.workers,
                                       pin_memory=dev == "cuda", drop_last=train and len(rows) > a.batch,
                                       persistent_workers=a.workers > 0)


dl_tr, dl_va, dl_te = loader(tr, True), loader(va, False), loader(te, False)

# ---------------------------------------------------------------- model
model = MultiTaskNet(IN_CH, backbone=a.backbone, pretrained=bool(a.pretrained), upscale=a.upscale).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
use_amp = dev == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
print(f"model {a.backbone}: {n_params(model) / 1e6:.2f}M parameters", flush=True)


def run(dl, train):
    model.train(train)
    tot, n, logits_all, reg_all, y_all, t_all, ids = 0.0, 0, [], [], [], [], []
    for b in dl:
        x = b["cube"].to(dev, non_blocking=True); y = b["cultivar"].to(dev); t = b["targets"].to(dev)
        with torch.set_grad_enabled(train), torch.autocast(dev, enabled=use_amp):
            lc, lr_ = model(x)
            loss = F.cross_entropy(lc, y) + F.mse_loss(lr_.float(), t)
        if train:
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tot += loss.item() * len(y); n += len(y)
        if not train:
            logits_all.append(lc.float().cpu()); reg_all.append(lr_.float().cpu())
            y_all.append(y.cpu()); t_all.append(t.cpu()); ids.append(b["image_id"])
    if train:
        return tot / n, None
    return tot / n, (torch.cat(logits_all), torch.cat(reg_all), torch.cat(y_all), torch.cat(t_all), torch.cat(ids))


best, best_ep, hist, t0 = 1e9, -1, [], time.time()
for ep in range(a.epochs):
    tl, _ = run(dl_tr, True)
    vl, _ = run(dl_va, False)
    sched.step()
    hist.append({"epoch": ep, "train_loss": tl, "val_loss": vl, "sec": round(time.time() - t0)})
    print(f"epoch {ep:2d} train {tl:.3f} val {vl:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if vl < best:
        best, best_ep = vl, ep
        torch.save(model.state_dict(), out / "model.pt")
    elif ep - best_ep >= a.patience:
        print("early stop"); break
pd.DataFrame(hist).to_csv(out / "history.csv", index=False)

# ---------------------------------------------------------------- test
model.load_state_dict(torch.load(out / "model.pt", map_location=dev))
_, (lc, lr_, y, t, ids) = run(dl_te, False)
pred_cls = lc.argmax(1).numpy()
mean, std = tstats
pred_reg = lr_.numpy() * std + mean
true_reg = t.numpy() * std + mean
metrics = {"split": a.split, "seed": a.seed, "fold": a.fold, "backbone": a.backbone, "best_epoch": best_ep,
           "upscale": a.upscale, "aug": a.aug, "camera": a.camera or "hyperspectral_204",
           "n_test": len(ids), "cultivar": classification_metrics(y.numpy(), pred_cls, 4)}
for j, name in enumerate(hyperleaf.REGRESSION_TARGETS):
    metrics[name] = r2_rmse(true_reg[:, j], pred_reg[:, j])
json.dump(metrics, open(out / "metrics.json", "w"), indent=1)
pred = pd.DataFrame({"ImageId": ids.numpy(), "cultivar_true": y.numpy(), "cultivar_pred": pred_cls})
for j, name in enumerate(hyperleaf.REGRESSION_TARGETS):
    pred[f"{name}_true"] = true_reg[:, j]; pred[f"{name}_pred"] = pred_reg[:, j]
pred.to_csv(out / "predictions.csv", index=False)
json.dump(vars(a), open(out / "config.json", "w"), indent=1, default=str)
print(json.dumps(metrics, indent=1))
print("saved to", out)
