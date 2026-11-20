"""Step 8 / E6: real-camera test on the Estrada leaves (avocado, olive, grape; 5 drying stages).

Training data: ONLY the ASD leaf spectra (400-1000 nm, resampled to 5 nm = 121 bands).
Test data: (a) the same test leaves simulated as 7 cameras, (b) the REAL MicaSense RedEdge images of the
test leaves (per-leaf band means; uncalibrated digital numbers turned into reflectance by a per-band
linear "empirical line" fitted on the TRAINING leaves only — like a reflectance panel; no trait labels used).

Tasks: crop (3 classes), drying stage (5 classes), chlorophyll and water content (FMC, fresh basis).
Models (all small, CPU):
  full        MLP on the 121-band spectrum; camera data interpolated back to 121 bands (today's workaround)
  svct        same MLP trained with random virtual cameras (held-out rule: the 7 real cameras never used)
  svct_tok    wavelength-aware set model (one token per band: value + centre + width), trained with SVCT
  real_ms     MLP trained on the REAL MicaSense bands of the training leaves (needs labelled drone data)
  sim_ms      MLP trained on simulated MicaSense bands (one model per camera, simulated)
Leaf-grouped 5-fold CV x 3 seeds.

  python experiments/estrada_e6.py --data data/raw/estrada_leaves/DATASET \
      --micasense results/step04_simulator_check/estrada_micasense_means.csv --out results/step08_estrada
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.cameras import list_cameras, load_camera  # noqa: E402
from anycam.camera_tf import back_to_full_matrix  # noqa: E402
from anycam.data import estrada  # noqa: E402
from anycam.data.group_splits import grouped_cv  # noqa: E402
from anycam.simulate import srf_matrix  # noqa: E402
from anycam.svct import SpectralStats, fourier_features, sample_structured_camera  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--micasense", required=True)
p.add_argument("--out", default="results/step08_estrada")
p.add_argument("--epochs", type=int, default=150)
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--folds", type=int, default=5)
p.add_argument("--radio", type=float, default=0.5)
p.add_argument("--quick", action="store_true", help="1 seed, 2 folds, 20 epochs")
a = p.parse_args()
if a.quick:
    a.seeds, a.epochs = [0], 20
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

# ---------------------------------------------------------------- data
WL = np.arange(400.0, 1001.0, 5.0)                                     # 121 bands
tab, spec = estrada.load(a.data)
idx = ((WL - 350).astype(int)[:, None] + np.arange(-2, 3)[None]).clip(0, spec.shape[1] - 1)
X = spec[:, idx].mean(2).astype(np.float64)                             # (N, 121), 5-nm averages
ms = pd.read_csv(a.micasense)
mapping = {"blue": "b1_mean", "green": "b2_mean", "red": "b3_mean", "red_edge": "b5_mean", "nir": "b4_mean"}
tab = tab.merge(ms, on=["crop", "leaf", "stage"], how="left")
MICA = load_camera("micasense_rededge_mx")
DN = tab[[mapping[b.name] for b in MICA.bands]].to_numpy(float)       # (N, 5) raw MicaSense means, camera band order
has_ms = np.isfinite(DN).all(1) & (DN > 0).all(1)
crop = tab.crop.map({c: i for i, c in enumerate(estrada.CROPS)}).to_numpy()
stage = tab.stage.to_numpy()
chl = tab.chlorophyll.to_numpy(float); chl_ok = chl < 200                # one value of 900 is a recording error
fmc = tab.FMC_f.to_numpy(float)
REG = np.stack([np.where(chl_ok, chl, np.nan), fmc], 1)
print(f"{len(tab)} spectra, {len(set(tab.leaf_id))} leaves, real MicaSense for {has_ms.sum()}", flush=True)

REAL = {n: load_camera(n) for n in list_cameras()}
HELD = list(REAL.values())
FULL_STEP = 5.0


# ---------------------------------------------------------------- models
class MLP(nn.Module):
    def __init__(self, d_in, hidden=256, drop=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.crop, self.stage, self.reg = nn.Linear(hidden, 3), nn.Linear(hidden, 5), nn.Linear(hidden, 2)

    def forward(self, x, *_):
        h = self.body(x)
        return self.crop(h), self.stage(h), self.reg(h)


class TokenNet(nn.Module):
    """One token per band: (value, Fourier(centre), Fourier(width)) -> shared MLP -> mean+max pool."""
    def __init__(self, hidden=128, n_freq=8, drop=0.1):
        super().__init__()
        self.n_freq = n_freq
        d_tok = 1 + 2 * (2 * n_freq + 1)
        self.tok = nn.Sequential(nn.Linear(d_tok, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.body = nn.Sequential(nn.Linear(2 * hidden, 256), nn.GELU(), nn.Dropout(drop))
        self.crop, self.stage, self.reg = nn.Linear(256, 3), nn.Linear(256, 5), nn.Linear(256, 2)

    def forward(self, x, centers, fwhms):
        N, B = x.shape
        f = torch.cat([fourier_features((centers - 350.0) / 700.0, self.n_freq),
                       fourier_features(fwhms / 100.0, self.n_freq)], 1)            # (B, d)
        t = torch.cat([x[..., None], f[None].expand(N, B, -1)], 2)
        h = self.tok(t)
        h = self.body(torch.cat([h.mean(1), h.max(1).values], 1))
        return self.crop(h), self.stage(h), self.reg(h)


def full_camera():
    from anycam.cameras import CameraProfile
    return CameraProfile.from_lists("full_121", WL.tolist(), [FULL_STEP] * len(WL), type="hyperspectral")


FULL = full_camera()


def render(xr, cam, stats, mode, nominal=None, gen=None, radio=0.0):
    """xr: raw reflectance (N, 121) tensor -> model input.
    mode 'interp': camera bands mapped back to 121 and standardised with full-spectrum stats.
    mode 'tok': native bands standardised with the camera's own stats (from the 121-band covariance)."""
    nominal = nominal or cam
    M = torch.eye(len(WL)) if cam is FULL else torch.tensor(srf_matrix(WL, cam), dtype=torch.float32)
    y = xr @ M.T
    if radio > 0:
        N, B = y.shape
        on = (torch.rand(N, 1, generator=gen) < radio).float()
        g = 0.8 + 0.45 * torch.rand(N, B, generator=gen)
        o = 0.02 * (2 * torch.rand(N, B, generator=gen) - 1)
        snr = 20 + 30 * torch.rand(N, 1, generator=gen)
        yp = y * g + o
        yp = yp + torch.randn(yp.shape, generator=gen) * yp.pow(2).mean(1, keepdim=True).sqrt() / 10 ** (snr / 20)
        y = on * yp + (1 - on) * y
    return from_bands(y, nominal, stats, mode)


def from_bands(y, nominal, stats, mode):
    """Camera band values (N, B) in reflectance -> model input."""
    if mode == "interp":
        if nominal is not FULL:
            y = y @ torch.tensor(back_to_full_matrix(nominal, WL), dtype=torch.float32).T
        mean, std = stats.mean, np.sqrt(np.diag(stats.cov))
    else:
        Mn = np.eye(len(WL)) if nominal is FULL else srf_matrix(WL, nominal)
        mean, std = stats.for_matrix(Mn)
    return (y - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(std, dtype=torch.float32)


def loss_fn(o, yc, ys, yr):
    lc, ls, lr_ = o
    m = torch.isfinite(yr)
    reg = F.mse_loss(lr_[m], yr[m]) if m.any() else lr_.sum() * 0
    return F.cross_entropy(lc, yc) + F.cross_entropy(ls, ys) + reg


def metrics(o, yc, ys, yr_raw, rmean, rstd):
    lc, ls, lr_ = (t.detach() for t in o)
    r = lr_.numpy() * rstd + rmean
    res = {"crop_acc": float((lc.argmax(1).numpy() == yc).mean()), "stage_acc": float((ls.argmax(1).numpy() == ys).mean())}
    for j, name in enumerate(["chlorophyll", "water_FMC"]):
        m = np.isfinite(yr_raw[:, j])
        t, pr = yr_raw[m, j], r[m, j]
        res[f"{name}_R2"] = float(1 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return res


def train(model, make_batch, n, epochs, seed, batch=128):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 8:
                continue
            x, extra, yc, ys, yr = make_batch(b)
            loss = loss_fn(model(x, *extra), yc, ys, yr)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    model.eval()
    return model


# ---------------------------------------------------------------- CV
splits = grouped_cv(tab, "leaf_id", "crop", n_folds=a.folds, seeds=tuple(a.seeds))
if a.quick:
    splits = splits[:2]
rows, t0 = [], time.time()
for sp in splits:
    tr = np.array(sp["train"] + sp["val"]); te = np.array(sp["test"])     # no early stopping -> val joins train
    torch.manual_seed(sp["seed"] * 100 + sp["fold"])
    stats = SpectralStats.fit(X[i].astype(np.float32)[:, None, None] for i in tr)
    rmean, rstd = np.nanmean(REG[tr], 0), np.nanstd(REG[tr], 0)
    Xt = torch.tensor(X, dtype=torch.float32)
    YC, YS = torch.tensor(crop), torch.tensor(stage)
    YR = torch.tensor((REG - rmean) / rstd, dtype=torch.float32)
    gen = torch.Generator().manual_seed(sp["seed"] * 1000 + sp["fold"])
    cam_rng = np.random.default_rng(sp["seed"] * 1000 + sp["fold"])

    # empirical-line calibration of the real MicaSense: DN -> simulated reflectance, fitted on training leaves
    sim_ms = X @ srf_matrix(WL, MICA).T
    trm = tr[has_ms[tr]]
    coef = [np.polyfit(DN[trm, b], sim_ms[trm, b], 1) for b in range(5)]
    real_ms = np.stack([np.polyval(coef[b], DN[:, b]) for b in range(5)], 1)
    cal_r2 = [float(np.corrcoef(real_ms[te[has_ms[te]], b], sim_ms[te[has_ms[te]], b])[0, 1] ** 2) for b in range(5)]

    models = {}
    # full-spectrum model
    Xin = render(Xt, FULL, stats, "interp")
    models["full"] = train(MLP(len(WL)), lambda b: (Xin[tr[b]], (), YC[tr[b]], YS[tr[b]], YR[tr[b]]), len(tr), a.epochs, sp["seed"])

    # SVCT (interp MLP) and SVCT tokens: a fresh virtual camera per batch, full spectrum 30 % of the time
    def svct_batch(mode):
        def f(b):
            cam = FULL if cam_rng.random() < 0.3 else sample_structured_camera(cam_rng, held_out=HELD, weights=(0.45, 0.35, 0.2))
            x = render(Xt[tr[b]], cam, stats, mode, gen=gen, radio=a.radio)
            extra = () if mode == "interp" else (torch.tensor(cam.centers, dtype=torch.float32), torch.tensor(cam.fwhms, dtype=torch.float32))
            return x, extra, YC[tr[b]], YS[tr[b]], YR[tr[b]]
        return f
    models["svct"] = train(MLP(len(WL)), svct_batch("interp"), len(tr), a.epochs, sp["seed"])
    models["svct_tok"] = train(TokenNet(), svct_batch("tok"), len(tr), a.epochs, sp["seed"])

    # one model per camera: simulated MicaSense, and REAL MicaSense (needs labelled drone images)
    ms_mean, ms_std = sim_ms[tr].mean(0), sim_ms[tr].std(0)
    S_in = torch.tensor((sim_ms - ms_mean) / ms_std, dtype=torch.float32)
    R_in = torch.tensor((real_ms - ms_mean) / ms_std, dtype=torch.float32)
    models["sim_ms"] = train(MLP(5), lambda b: (S_in[tr[b]], (), YC[tr[b]], YS[tr[b]], YR[tr[b]]), len(tr), a.epochs, sp["seed"])
    models["real_ms"] = train(MLP(5), lambda b: (R_in[trm[b]], (), YC[trm[b]], YS[trm[b]], YR[trm[b]]), len(trm), a.epochs, sp["seed"])

    # ---------------- test
    tem = te[has_ms[te]]
    def ev(model, x, extra, rows_):
        with torch.no_grad():
            return metrics(model(x, *extra), crop[rows_], stage[rows_], REG[rows_], rmean, rstd)
    base = {"seed": sp["seed"], "fold": sp["fold"]}
    for name in ("full", "svct", "svct_tok"):
        mode = "tok" if name == "svct_tok" else "interp"
        m = models[name]
        tok = lambda cam: () if mode == "interp" else (torch.tensor(cam.centers, dtype=torch.float32), torch.tensor(cam.fwhms, dtype=torch.float32))
        rows.append({**base, "model": name, "test": "full_spectrum", **ev(m, render(Xt[te], FULL, stats, mode), tok(FULL), te)})
        for cname, cam in REAL.items():
            rows.append({**base, "model": name, "test": f"sim_{cname}", **ev(m, render(Xt[te], cam, stats, mode), tok(cam), te)})
        rx = from_bands(torch.tensor(real_ms[tem], dtype=torch.float32), MICA, stats, mode)
        rows.append({**base, "model": name, "test": "REAL_micasense", **ev(m, rx, tok(MICA), tem)})
    rows.append({**base, "model": "sim_ms", "test": "sim_micasense_rededge_mx", **ev(models["sim_ms"], S_in[te], (), te)})
    rows.append({**base, "model": "sim_ms", "test": "REAL_micasense", **ev(models["sim_ms"], R_in[tem], (), tem)})
    rows.append({**base, "model": "real_ms", "test": "REAL_micasense", **ev(models["real_ms"], R_in[tem], (), tem)})
    rows.append({**base, "model": "calibration", "test": "band_R2_real_vs_sim", **{f"band{b + 1}": cal_r2[b] for b in range(5)}})
    print(f"seed {sp['seed']} fold {sp['fold']} done ({time.time() - t0:.0f}s) | REAL MicaSense crop acc: "
          + " ".join(f"{r['model']} {r['crop_acc']:.2f}" for r in rows[-8:] if r["test"] == "REAL_micasense"), flush=True)

df = pd.DataFrame(rows)
df.to_csv(out / "estrada_e6_all_folds.csv", index=False)
cols = ["crop_acc", "stage_acc", "chlorophyll_R2", "water_FMC_R2"]
res = df[df.model != "calibration"].groupby(["test", "model"])[cols].agg(["mean", "std"]).round(3)
res.to_csv(out / "estrada_e6_summary.csv")
pd.set_option("display.width", 220)
key = df[df.test.isin(["full_spectrum", "sim_micasense_rededge_mx", "REAL_micasense"]) & (df.model != "calibration")]
print("\nMean over folds (std in brackets):")
summ = key.groupby(["test", "model"])[cols].agg(lambda v: f"{v.mean():.3f} ({v.std():.3f})")
print(summ.to_string())
print("\nAll simulated cameras, mean crop acc / stage acc / chlorophyll R2 / water R2:")
sim = df[df.test.str.startswith("sim_") & df.model.isin(["full", "svct", "svct_tok"])]
print(sim.groupby(["model"])[cols].mean().round(3).to_string())
print("\nCalibration quality (R2 real vs simulated band, test leaves):",
      df[df.model == "calibration"][[f"band{b}" for b in range(1, 6)]].mean().round(3).to_dict())
json.dump(vars(a), open(out / "config.json", "w"), indent=1)
print("saved", out)
