"""Figures for the AnyCam paper draft. python paper/make_figures.py  (writes paper/figures/*.pdf)"""
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from anycam.cameras import load_camera  # noqa: E402
from anycam.simulate import gaussian_srf  # noqa: E402

OUT = ROOT / "paper/figures"; OUT.mkdir(parents=True, exist_ok=True)
# documented categorical palette (fixed order) + text inks
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#d9d8d3"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "axes.axisbelow": True,
                     "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight"})
CAMS = ["generic_rgb", "generic_rgb_nir", "dji_p4_multispectral", "micasense_rededge_mx", "micasense_no_red_edge",
        "parrot_sequoia", "sentinel2a_vnir"]
SHORT = {"generic_rgb": "Phone RGB\n(3)", "generic_rgb_nir": "RGB+NIR\n(4)", "dji_p4_multispectral": "DJI P4M\n(5)",
         "micasense_rededge_mx": "MicaSense\n(5)", "micasense_no_red_edge": "MicaSense\nno RE (4)",
         "parrot_sequoia": "Sequoia\n(4)", "sentinel2a_vnir": "Sentinel-2\n(10)"}


def bars(ax, groups, series, colors, labels, hatches=None, ylim=None, ylabel=""):
    n = len(series); w = 0.8 / n; x = np.arange(len(groups))
    for k, (vals, c, lab) in enumerate(zip(series, colors, labels)):
        v = np.asarray(vals, float)
        ax.bar(x + (k - (n - 1) / 2) * w, v, w * 0.92, color=c, label=lab, edgecolor="white", linewidth=0.6,
               hatch=(hatches[k] if hatches else None))
    ax.set_xticks(x); ax.set_xticklabels(groups); ax.axhline(0, color=INK2, lw=0.6)
    ax.grid(axis="x", visible=False)
    if ylim: ax.set_ylim(*ylim)
    ax.set_ylabel(ylabel)


# ---------------------------------------------------------------- Fig 1: method overview
def fig_method():
    fig, ax = plt.subplots(figsize=(7.4, 2.9)); ax.set_xlim(0, 100); ax.set_ylim(-2, 36); ax.axis("off")

    def box(x, y, w, h, title, body, fc="#f3f2ee", ec=INK2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.2", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=7.6, weight="bold", color=INK)
        ax.text(x + w / 2, y + h - 6.6, body, ha="center", va="top", fontsize=6.5, color=INK2, linespacing=1.35)

    def arrow(x0, x1, y=19):
        ax.annotate("", (x1, y), (x0, y), arrowprops=dict(arrowstyle="-|>", color=INK2, lw=0.9))
    box(1, 6, 16, 26, "Hyperspectral\ntraining data", "204 bands\n400–1000 nm\n(one camera,\nlabelled once)")
    box(21, 6, 22, 26, "Virtual camera\nper mini-batch", "3–12 bands, random\ncentres & widths;\nRGB-like / multispectral /\nrandom families;\n30 % full 204 bands;\n7 real cameras held out")
    box(47.5, 6, 17, 26, "Calibration\nerrors", "gain 0.8–1.25\noffset ±0.02\nnoise 20–50 dB\n(p = 0.5)")
    box(69.5, 18.5, 29, 13.5, "(a) wavelength-conditioned\nstem", "filters from each band's centre\nand width (ImageNet RGB init)", fc="#e9f1fb", ec=S1)
    box(69.5, 3.5, 29, 13.5, "(b) interpolate back\nto 204 bands", "plain 204-band ResNet18;\nno architecture change", fc="#fdeee7", ec=S2)
    arrow(17.8, 20.4); arrow(43.8, 46.8); arrow(65.3, 68.8, 26); arrow(65.3, 68.8, 11)
    ax.text(84, -1.6, "→ ResNet18 → cultivar + 4 trait regressions", ha="center", fontsize=7.2, color=INK)
    fig.savefig(OUT / "fig1_method.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 2: camera profiles on a leaf spectrum
def fig_cameras():
    from scipy.io import loadmat
    mat = ROOT.parent / "x"  # placeholder
    f = pathlib.Path("/mnt/user-data/uploads/Assignments/anycam/data/raw/estrada_leaves/DATASET/Avocado/Spectral_avocado.mat")
    wl = np.arange(350, 2501)
    spec = None
    if f.exists():
        m = {k: np.asarray(v) for k, v in loadmat(f).items() if not k.startswith("__")}
        arr = [v for v in m.values() if v.ndim == 2 and v.shape[1] == 2151]
        spec = arr[0].mean(0) if arr else None
    lam = np.linspace(400, 1000, 601)
    fig, axes = plt.subplots(len(CAMS), 1, figsize=(7.2, 5.0), sharex=True)
    for ax, c in zip(axes, CAMS):
        cam = load_camera(c)
        if spec is not None:
            s = np.interp(lam, wl, spec)
            ax.plot(lam, s / s.max(), color=INK2, lw=1.0)
        for b in cam.bands:
            g = gaussian_srf(lam, b.center, b.fwhm); g = g / g.max()
            ax.fill_between(lam, 0, g, color=S1, alpha=0.35, lw=0); ax.plot(lam, g, color=S1, lw=0.8)
        ax.set_yticks([]); ax.set_ylim(0, 1.08); ax.grid(False); ax.spines["left"].set_visible(False)
        ax.text(1.005, 0.5, SHORT[c].replace("\n", " "), transform=ax.transAxes, va="center", fontsize=7.5, color=INK)
    axes[-1].set_xlabel("Wavelength (nm)"); axes[-1].set_xlim(400, 1000)
    axes[0].set_title("Held-out camera profiles (blue: band responses) over a mean leaf reflectance spectrum (grey)",
                      fontsize=8, color=INK, loc="left")
    fig.savefig(OUT / "fig2_cameras.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 3: per-camera results (Step 7)
def fig_per_camera():
    t = pd.read_csv(ROOT / "paper/data/per_camera.csv"); t = t[t.condition == "nominal"]
    M = [("baseline", "204-band model"), ("svct_stem", "SVCT + wavelength stem"),
         ("svct_interp", "SVCT + interpolation"), ("oracle", "one model per camera")]
    fig, axes = plt.subplots(2, 1, figsize=(7.2, 4.6), sharex=True)
    for ax, col, yl, lim in [(axes[0], "cultivar_acc", "Cultivar accuracy", (0, 1)),
                             (axes[1], "Fertilizer_R2", "Fertilizer R²", (-1.5, 1))]:
        ser = [[t[(t.camera == c) & (t.method == m)][col].iloc[0] for c in CAMS] for m, _ in M]
        bars(ax, [SHORT[c] for c in CAMS], ser, [S1, S2, S3, S4], [l for _, l in M],
             hatches=[None, None, None, "////"], ylim=lim, ylabel=yl)
    axes[0].axhline(0.25, color=INK2, lw=0.6, ls=":"); axes[0].text(6.45, 0.26, "chance", fontsize=6.5, color=INK2, ha="right")
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7.3)
    fig.savefig(OUT / "fig3_per_camera.pdf"); plt.close(fig)


# ---------------------------------------------------------------- Fig 4: real second camera (Step 8)
def fig_real():
    d = pd.read_csv(ROOT / "results/step08_deephs/deephs_e4_all_folds.csv")
    e = pd.read_csv(ROOT / "results/step08_estrada/estrada_e6_all_folds.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), gridspec_kw={"width_ratios": [1, 1]})
    dd = d[d.test == "REAL Corning (full)"].groupby("model")[["fruit_acc", "ripeness_acc", "sugar_R2", "log_firmness_R2"]].mean()
    tasks = ["fruit_acc", "ripeness_acc", "sugar_R2", "log_firmness_R2"]
    bars(axes[0], ["Fruit\n(acc.)", "Ripeness\n(acc.)", "Sugar\n(R²)", "Firmness\n(R²)"],
         [dd.loc[m, tasks] for m in ("full", "svct", "corning_oracle")], [S1, S2, S4],
         ["trained on FX10 only", "SVCT, trained on FX10 only", "trained on Corning"], hatches=[None, None, "////"],
         ylim=(-0.5, 1.05))
    axes[0].set_title("(a) Fruit: FX10 → real Corning camera", fontsize=8, loc="left", color=INK)
    ee = e[e.test == "REAL_micasense"].groupby("model")[["crop_acc", "stage_acc", "chlorophyll_R2", "water_FMC_R2"]].mean()
    t2 = ["crop_acc", "stage_acc", "chlorophyll_R2", "water_FMC_R2"]
    bars(axes[1], ["Crop\n(acc.)", "Stage\n(acc.)", "Chloroph.\n(R²)", "Water\n(R²)"],
         [ee.loc[m, t2] for m in ("full", "svct_tok", "real_ms")], [S1, S2, S4],
         ["trained on spectrometer only", "SVCT", "trained on real MicaSense"], hatches=[None, None, "////"],
         ylim=(-0.5, 1.05))
    axes[1].set_title("(b) Leaves: spectrometer → real MicaSense", fontsize=8, loc="left", color=INK)
    for ax in axes:
        ax.legend(fontsize=6.6, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=1)
    fig.tight_layout(); fig.savefig(OUT / "fig4_real_cameras.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_method(); fig_cameras(); fig_per_camera(); fig_real()
    for f in sorted(OUT.glob("*.pdf")):
        print(f)
