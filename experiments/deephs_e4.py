"""Step 8 / E4: real cross-camera test on DeepHS fruit (kaki, mango, papaya).

Every fruit was recorded by TWO real hyperspectral cameras: Specim FX10 (224 bands, 398-1004 nm) and
Corning microHSI 410 (249 bands, 408-901 nm; different optics, range and radiometry).
Models are trained ONLY on FX10 spectra of the training fruits and tested on the REAL Corning recordings of
unseen fruits (plus both cameras reduced to the 7 simulated camera profiles).
Input per recording: median spectrum of the central fruit area (results/step04_simulator_check/deephs_mean_spectra.csv).
Tasks: fruit type (3), ripeness (unripe / perfect / overripe), sugar (Brix), firmness (log).
Models: full (plain MLP, camera interpolated to the FX10 grid) | svct (same MLP, SVCT) | svct_tok (wavelength tokens, SVCT)
        | corning_oracle (MLP trained on labelled Corning recordings: "one model per camera").
Fruit-grouped 5-fold CV x 3 seeds.  python experiments/deephs_e4.py --out results/step08_deephs
"""
import argparse
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
from anycam.data import deephs  # noqa: E402
from anycam.data.group_splits import grouped_cv  # noqa: E402
from anycam.spectral1d import (MLP, Spectral1D, TokenNet, cam_tensors, evaluate, fit, grid_camera,  # noqa: E402
                               svct_camera)

p = argparse.ArgumentParser()
p.add_argument("--labels", default=str(ROOT / "data/raw/deephs_fruit/labels"))
p.add_argument("--spectra", default=str(ROOT / "results/step04_simulator_check/deephs_mean_spectra.csv"))
p.add_argument("--out", default="results/step08_deephs")
p.add_argument("--epochs", type=int, default=300)
p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
p.add_argument("--radio", type=float, default=0.5)
p.add_argument("--quick", action="store_true")
a = p.parse_args()
if a.quick:
    a.seeds, a.epochs = [0], 30
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
torch.set_num_threads(2)

lab, WLS = deephs.load_labels(a.labels)
sp = pd.read_csv(a.spectra)
sp["vec"] = sp.spectrum.map(lambda s: np.array(s.split(), float))
df = sp.drop(columns="spectrum").merge(lab[["data_file", "ripeness_idx", "sugar", "firmness"]], on="data_file")
WL_FX, WL_CO = WLS["VIS"], WLS["VIS_COR"]
FRUITS = list(deephs.FRUITS)
# one row per (fruit_id, side) with both cameras
fx = df[df.camera == "VIS"].set_index(["fruit_id", "side"]); co = df[df.camera == "VIS_COR"].set_index(["fruit_id", "side"])
pairs = fx.join(co[["vec"]], rsuffix="_co", how="inner").reset_index()
XF = np.stack(pairs.vec); XC = np.stack(pairs.vec_co)
YF = pairs.fruit.map({f: i for i, f in enumerate(FRUITS)}).to_numpy()
YR = pairs.ripeness_idx.to_numpy()
REG = np.stack([pairs.sugar.to_numpy(float), np.log1p(pairs.firmness.to_numpy(float))], 1)
print(f"{len(pairs)} paired recordings, {pairs.fruit_id.nunique()} fruits; FX10 {XF.shape[1]} bands, Corning {XC.shape[1]} bands",
      flush=True)
CLS, REGN = ["fruit", "ripeness"], ["sugar", "log_firmness"]
HEADS = [3, 3, 2]
CORNING = grid_camera(WL_CO, "corning_microhsi410")
REAL = {n: load_camera(n) for n in list_cameras()}
HELD = list(REAL.values())


def cams_for(src_wl):
    ok = {}
    for n, c in REAL.items():
        try:
            from anycam.simulate import srf_matrix
            srf_matrix(src_wl, c)
            ok[n] = c
        except ValueError:
            pass
    return ok


CAMS_FX, CAMS_CO = cams_for(WL_FX), cams_for(WL_CO)
splits = grouped_cv(pairs.assign(r=pairs.ripeness_idx), "fruit_id", "r", n_folds=5, seeds=tuple(a.seeds))
if a.quick:
    splits = splits[:2]
rows, t0 = [], time.time()
Xf, Xc = torch.tensor(XF, dtype=torch.float32), torch.tensor(XC, dtype=torch.float32)
for spl in splits:
    tr = np.array(spl["train"] + spl["val"]); te = np.array(spl["test"])
    seed = spl["seed"] * 100 + spl["fold"]
    torch.manual_seed(seed)
    S = Spectral1D(WL_FX, XF[tr])
    rmean, rstd = REG[tr].mean(0), REG[tr].std(0)
    yc = [torch.tensor(YF), torch.tensor(YR)]
    yr = torch.tensor((REG - rmean) / rstd, dtype=torch.float32)
    gen = torch.Generator().manual_seed(seed); cam_rng = np.random.default_rng(seed)

    def batch_from(feat_fn, rows_):
        def f(b):
            x, extra = feat_fn(rows_[b])
            return x, extra, [c[rows_[b]] for c in yc], yr[rows_[b]]
        return f

    full_in = S.model_input(Xf, S.full, "interp")
    models = {"full": fit(MLP(len(WL_FX), HEADS), batch_from(lambda r: (full_in[r], ()), tr), len(tr), a.epochs, seed, 2)}

    def svct_feats(mode):
        def f(r):
            cam = svct_camera(cam_rng, S.full, HELD)
            x = S.model_input(S.bands(Xf[r], cam, radio=a.radio, gen=gen), cam, mode)
            return x, (() if mode == "interp" else cam_tensors(cam))
        return f
    models["svct"] = fit(MLP(len(WL_FX), HEADS), batch_from(svct_feats("interp"), tr), len(tr), a.epochs, seed, 2)
    models["svct_tok"] = fit(TokenNet(HEADS), batch_from(svct_feats("tok"), tr), len(tr), a.epochs, seed, 2)
    Sc = Spectral1D(WL_CO, XC[tr])
    co_in = Sc.model_input(Xc, Sc.full, "interp")
    models["corning_oracle"] = fit(MLP(len(WL_CO), HEADS), batch_from(lambda r: (co_in[r], ()), tr), len(tr), a.epochs, seed, 2)

    def ev(name, x, extra):
        with torch.no_grad():
            return evaluate(models[name](x, *extra), [YF[te], YR[te]], REG[te], rmean, rstd, CLS, REGN)
    base = {"seed": spl["seed"], "fold": spl["fold"]}
    for name in ("full", "svct", "svct_tok"):
        mode = "tok" if name == "svct_tok" else "interp"
        ex = lambda cam: () if mode == "interp" else cam_tensors(cam)
        rows.append({**base, "model": name, "test": "FX10 (same camera)",
                     **ev(name, S.model_input(Xf[te], S.full, mode), ex(S.full))})
        rows.append({**base, "model": name, "test": "REAL Corning (full)",
                     **ev(name, S.model_input(Xc[te], CORNING, mode), ex(CORNING))})
        for cn, cam in CAMS_CO.items():
            y = S.bands(Xc[te], cam, src_wl=WL_CO)
            rows.append({**base, "model": name, "test": f"REAL Corning -> {cn}", **ev(name, S.model_input(y, cam, mode), ex(cam))})
        for cn, cam in CAMS_FX.items():
            y = S.bands(Xf[te], cam)
            rows.append({**base, "model": name, "test": f"FX10 -> {cn}", **ev(name, S.model_input(y, cam, mode), ex(cam))})
    rows.append({**base, "model": "corning_oracle", "test": "REAL Corning (full)",
                 **ev("corning_oracle", co_in[te], ())})
    r = [x for x in rows[-40:] if x["test"] == "REAL Corning (full)"]
    print(f"seed {spl['seed']} fold {spl['fold']} ({time.time() - t0:.0f}s) | REAL Corning ripeness: "
          + " ".join(f"{x['model']} {x['ripeness_acc']:.2f}" for x in r), flush=True)

res = pd.DataFrame(rows)
res.to_csv(out / "deephs_e4_all_folds.csv", index=False)
cols = [f"{c}_acc" for c in CLS] + [f"{r}_R2" for r in REGN]
pd.set_option("display.width", 220)
fmt = lambda v: f"{v.mean():.3f} ({v.std():.3f})"
key = res[res.test.isin(["FX10 (same camera)", "REAL Corning (full)"])]
print("\nMean (std) over folds:")
print(key.groupby(["test", "model"])[cols].agg(fmt).to_string())
for prefix in ("REAL Corning -> ", "FX10 -> "):
    sub = res[res.test.str.startswith(prefix)]
    print(f"\n{prefix.strip(' ->')} reduced to the simulated camera profiles (mean over cameras and folds):")
    print(sub.groupby("model")[cols].mean().round(3).to_string())
res.groupby(["test", "model"])[cols].agg(["mean", "std"]).round(3).to_csv(out / "deephs_e4_summary.csv")
json.dump(vars(a), open(out / "config.json", "w"), indent=1)
print("saved", out)
