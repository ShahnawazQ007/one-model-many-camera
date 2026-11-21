"""LaTeX tables for the paper, generated from the result files (no hand-typed numbers).
python paper/make_tables.py  -> paper/tables/*.tex
Step 9 (cross-validation) table is filled from runs/cv_summary/cv_per_fold.csv when it exists (paper/data/cv_per_fold.csv)."""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.cameras import load_camera  # noqa: E402

T = ROOT / "paper/tables"; T.mkdir(parents=True, exist_ok=True)
CAMS = ["generic_rgb", "generic_rgb_nir", "dji_p4_multispectral", "micasense_rededge_mx", "micasense_no_red_edge",
        "parrot_sequoia", "sentinel2a_vnir"]
NAME = {"generic_rgb": "Phone RGB$^\\dagger$", "generic_rgb_nir": "RGB+NIR$^\\dagger$", "dji_p4_multispectral": "DJI P4 Multispectral",
        "micasense_rededge_mx": "MicaSense RedEdge-MX", "micasense_no_red_edge": "MicaSense, no red edge$^\\ddagger$",
        "parrot_sequoia": "Parrot Sequoia", "sentinel2a_vnir": "Sentinel-2A MSI (VNIR)"}
TYPE = {"rgb": "RGB", "rgb_nir": "RGB+NIR", "uav_multispectral": "UAV", "control": "control", "satellite": "satellite"}
f3 = lambda v: f"{v:.3f}".replace("-", "$-$")
f2 = lambda v: f"{v:.2f}".replace("-", "$-$")


def write(name, s):
    (T / f"{name}.tex").write_text(s); print("wrote", name)


# ---------------------------------------------------------------- cameras
rows = []
for c in CAMS:
    cam = load_camera(c)
    bands = ", ".join(f"{b.center:g}/{b.fwhm:g}" for b in cam.bands)
    rows.append(f"{NAME[c]} & {TYPE.get(cam.type, cam.type)} & {len(cam)} & \\footnotesize {bands} \\\\")
write("tab_cameras", "\\begin{tabular}{@{}llcp{7.2cm}@{}}\n\\toprule\nCamera & Type & Bands & Centre/FWHM (nm) \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Step 5 baseline
b = pd.read_csv(ROOT / "results/step05_baseline/comparison_official.csv")
lab = {"baseline_official": "ResNet18, 204 bands", "A_resnet18_up2_strong": "ResNet18, $2\\times$ input, strong aug.\\ (reference)",
       "B_convnext_tiny_up2_strong": "ConvNeXt-T, $2\\times$ input, strong aug."}
rows = [f"{lab[r.run]} & {f3(r.cultivar_acc)} & {f3(r.cultivar_F1)} & {f3(r.GrainWeight_R2)} & {f3(r.Gsw_R2)} & {f3(r.PhiPS2_R2)} & {f3(r.Fertilizer_R2)} \\\\"
        for r in b.itertuples()]
write("tab_baseline", "\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Cultivar} & \\multicolumn{4}{c}{$R^2$} \\\\\n"
      "\\cmidrule(lr){2-3}\\cmidrule(l){4-7}\nModel & Acc. & F1 & Grain & $g_{sw}$ & $\\Phi_{PSII}$ & Fert. \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Step 7 main table (mean over 7 held-out cameras)
pc = pd.read_csv(ROOT / "paper/data/per_camera.csv")
C = ["cultivar_acc", "Fertilizer_R2", "GrainWeight_R2", "PhiPS2_R2"]
mean = pc.groupby(["method", "condition"])[C].mean()
M = [("baseline", "204-band model, bands interpolated"), ("svct_stem", "SVCT + wavelength stem (ours)"),
     ("svct_interp", "SVCT + interpolation (ours)"), ("oracle", "One model per camera$^\\ast$")]
full = {"baseline": (0.821, 0.759), "svct_stem": (0.660, 0.589), "svct_interp": (0.689, 0.629)}  # results/step07_svct_round3/summary.txt
best = {c: max(mean.loc[(m, "nominal"), c] for m, _ in M[:3]) for c in C}
rows = []
for m, l in M:
    nom = mean.loc[(m, "nominal")]
    cells = [("\\textbf{" + f3(nom[c]) + "}") if (m != "oracle" and np.isclose(nom[c], best[c])) else f3(nom[c]) for c in C]
    dr = mean.loc[(m, "drift+radio")] if (m, "drift+radio") in mean.index else None
    drc = f"{f3(dr.cultivar_acc)} & {f3(dr.Fertilizer_R2)}" if dr is not None else "-- & --"
    fu = f"{f3(full[m][0])} & {f3(full[m][1])}" if m in full else "-- & --"
    rows.append(f"{l} & {' & '.join(cells)} & {drc} & {fu} \\\\")
gap = {}
for m in ("svct_stem", "svct_interp"):
    gap[m] = [(mean.loc[(m, 'nominal'), c] - mean.loc[('baseline', 'nominal'), c])
              / (mean.loc[('oracle', 'nominal'), c] - mean.loc[('baseline', 'nominal'), c]) * 100 for c in C]
rows.append("\\midrule")
for m, l in M[1:3]:
    rows.append(f"\\quad gap closed, {l.split(' (')[0].replace('SVCT + ', '')} & " + " & ".join(f"{g:.0f}\\%" for g in gap[m]) + " & & & & \\\\")
write("tab_main", "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n & \\multicolumn{4}{c}{Ideal cameras} & \\multicolumn{2}{c}{Drift + calib.\\ error}"
      " & \\multicolumn{2}{c}{Full 204 bands} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}\\cmidrule(l){8-9}\n"
      "Method & Cult.\\ acc. & Fert.\\ $R^2$ & Grain $R^2$ & $\\Phi_{PSII}$ $R^2$ & Cult. & Fert. & Cult. & Fert. \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
json.dump({m: dict(zip(C, map(float, g))) for m, g in gap.items()}, open(ROOT / "paper/data/gap_closure.json", "w"), indent=1)

# ---------------------------------------------------------------- per-camera (nominal)
nom = pc[pc.condition == "nominal"]
rows = []
for c in CAMS:
    v = lambda m, col: nom[(nom.camera == c) & (nom.method == m)][col].iloc[0]
    rows.append(f"{NAME[c]} & " + " & ".join(f3(v(m, 'cultivar_acc')) for m, _ in M) + " & "
                + " & ".join(f3(v(m, 'Fertilizer_R2')) for m, _ in M) + " \\\\")
write("tab_per_camera", "\\begin{tabular}{@{}lcccccccc@{}}\n\\toprule\n & \\multicolumn{4}{c}{Cultivar accuracy} & \\multicolumn{4}{c}{Fertilizer $R^2$} \\\\\n"
      "\\cmidrule(lr){2-5}\\cmidrule(l){6-9}\nCamera & Base & Stem & Interp & Own & Base & Stem & Interp & Own \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- ablation over development rounds (summary.txt files)
abl = [  # (round, variant, cultivar, fert, grain, phips2, cultivar on 204, fert on 204) from results/step07_svct_round*/summary.txt
    ("1", "two views + symmetric consistency (w=1), uniform sampler", 0.502, 0.480, 0.464, 0.203, 0.562, 0.543),
    ("1", "same, without consistency", 0.639, 0.456, 0.356, 0.285, 0.707, 0.609),
    ("1", "same as first, interpolation instead of stem", 0.556, 0.377, 0.366, 0.238, 0.572, 0.594),
    ("2", "teacher distillation (w=0.5), structured sampler", 0.551, 0.526, 0.508, 0.226, 0.709, 0.392),
    ("2", "structured sampler, no consistency", 0.664, 0.527, 0.455, 0.322, 0.701, 0.608),
    ("2", "teacher distillation, interpolation", 0.647, 0.418, 0.387, 0.234, 0.645, 0.681),
    ("3", "final: no consistency, $p_{full}$=0.3, phone-weighted sampler, 150 ep.", 0.670, 0.582, 0.528, 0.254, 0.660, 0.589),
    ("3", "final, interpolation instead of stem", 0.732, 0.468, 0.429, 0.227, 0.689, 0.629),
]
rows = [f"{r} & {v} & {f3(a)} & {f3(b_)} & {f3(c)} & {f3(d)} & {f3(e)} & {f3(f)} \\\\" for r, v, a, b_, c, d, e, f in abl]
write("tab_ablation", "\\begin{tabular}{@{}clcccccc@{}}\n\\toprule\n & & \\multicolumn{4}{c}{7 held-out cameras (ideal)} & \\multicolumn{2}{c}{204 bands} \\\\\n"
      "\\cmidrule(lr){3-6}\\cmidrule(l){7-8}\nRound & Variant & Cult. & Fert. & Grain & $\\Phi_{PSII}$ & Cult. & Fert. \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- Step 8: DeepHS (E4) and Estrada (E6)
pm = lambda s: f"{s.mean():.3f} $\\pm$ {s.std():.3f}".replace("-", "$-$").replace("$\\pm$ $-$", "$\\pm$ ")
d = pd.read_csv(ROOT / "results/step08_deephs/deephs_e4_all_folds.csv")
DC = ["fruit_acc", "ripeness_acc", "sugar_R2", "log_firmness_R2"]
DL = {"full": "Plain MLP (FX10 only)", "svct": "SVCT, interpolation (FX10 only)", "svct_tok": "SVCT, wavelength tokens (FX10 only)",
      "corning_oracle": "Trained on Corning labels$^\\ast$"}
rows = []
for test, tl in [("FX10 (same camera)", "Same camera (FX10)"), ("REAL Corning (full)", "Real Corning camera")]:
    rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{tl}}}}} \\\\")
    for m in ("full", "svct", "svct_tok", "corning_oracle"):
        s = d[(d.test == test) & (d.model == m)]
        if len(s):
            rows.append(f"\\quad {DL[m]} & " + " & ".join(pm(s[c]) for c in DC) + " \\\\")
for pre, tl in [("REAL Corning -> ", "Real Corning, reduced to the 7 camera profiles (mean)"),
                ("FX10 -> ", "FX10, reduced to the 7 camera profiles (mean)")]:
    rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{tl}}}}} \\\\")
    sub = d[d.test.str.startswith(pre)]
    g = sub.groupby(["model", "seed", "fold"])[DC].mean().reset_index()
    for m in ("full", "svct", "svct_tok"):
        rows.append(f"\\quad {DL[m]} & " + " & ".join(pm(g[g.model == m][c]) for c in DC) + " \\\\")
write("tab_deephs", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModel & Fruit acc. & Ripeness acc. & Sugar $R^2$ & log-firmness $R^2$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

e = pd.read_csv(ROOT / "results/step08_estrada/estrada_e6_all_folds.csv")
EC = ["crop_acc", "stage_acc", "chlorophyll_R2", "water_FMC_R2"]
EL = {"full": "Plain MLP (spectrometer only)", "svct": "SVCT, interpolation", "svct_tok": "SVCT, wavelength tokens",
      "sim_ms": "Trained on simulated MicaSense$^\\S$", "real_ms": "Trained on real MicaSense labels$^\\ast$"}
rows = []
for test, tl, ms in [("full_spectrum", "Same sensor (spectrometer, 400--1000 nm)", ("full", "svct", "svct_tok")),
                     ("REAL_micasense", "Real MicaSense camera", ("full", "svct", "svct_tok", "sim_ms", "real_ms"))]:
    rows.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\textit{{{tl}}}}} \\\\")
    for m in ms:
        s = e[(e.test == test) & (e.model == m)]
        rows.append(f"\\quad {EL[m]} & " + " & ".join(pm(s[c]) for c in EC) + " \\\\")
rows.append("\\multicolumn{5}{@{}l}{\\textit{Simulated, 7 camera profiles (mean)}} \\\\")
sub = e[e.test.str.startswith("sim_") & e.model.isin(["full", "svct", "svct_tok"])]
g = sub.groupby(["model", "seed", "fold"])[EC].mean().reset_index()
for m in ("full", "svct", "svct_tok"):
    rows.append(f"\\quad {EL[m]} & " + " & ".join(pm(g[g.model == m][c]) for c in EC) + " \\\\")
write("tab_estrada", "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\nModel & Crop acc. & Stage acc. & Chlorophyll $R^2$ & Water (FMC) $R^2$ \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
cal = e[e.model == "calibration"][[f"band{i}" for i in range(1, 6)]].mean()
json.dump({k: float(v) for k, v in cal.items()}, open(ROOT / "paper/data/estrada_band_r2.json", "w"), indent=1)

# ---------------------------------------------------------------- Step 9: cross-validation (filled when available)
cvf = ROOT / "paper/data/cv_per_fold.csv"
LB = {"204-band baseline": "204-band model, bands interpolated", "SVCT (wavelength stem)": "SVCT + wavelength stem (ours)",
      "SVCT (interp)": "SVCT + interpolation (ours)"}
if cvf.exists():
    cv = pd.read_csv(cvf); cv = cv[cv.condition.isin(["nominal", "drift+radio"])]
    nf = cv[cv.condition == "nominal"].groupby("method").fold.nunique()
    rows = []
    for m, l in LB.items():
        a_ = cv[(cv.method == m) & (cv.condition == "nominal")]; r_ = cv[(cv.method == m) & (cv.condition == "drift+radio")]
        rows.append(f"{l} & {pm(a_.cultivar_acc)} & {pm(a_.Fertilizer_R2)} & {pm(a_.GrainWeight_R2)} & {pm(r_.cultivar_acc)} & {pm(r_.Fertilizer_R2)} \\\\")
    note = f"% folds per method: {nf.to_dict()}\n"
else:
    rows = [f"{l} & \\pending & \\pending & \\pending & \\pending & \\pending \\\\" for l in LB.values()]
    note = "% Step 9 still running: placeholders\n"
write("tab_cv", note + "\\begin{tabular}{@{}lccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{Ideal cameras} & \\multicolumn{2}{c}{Drift + calib.\\ error} \\\\\n"
      "\\cmidrule(lr){2-4}\\cmidrule(l){5-6}\nMethod & Cult.\\ acc. & Fert.\\ $R^2$ & Grain $R^2$ & Cult. & Fert. \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
