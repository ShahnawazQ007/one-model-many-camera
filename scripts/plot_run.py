"""Figures for one training run folder: training curves, confusion matrix, predicted vs true.

Usage: python scripts/plot_run.py runs/baseline_official [more run folders...]
Writes figures.png into each run folder.
"""
import json
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam.data.hyperleaf import CULTIVARS, REGRESSION_TARGETS  # noqa: E402

NAMES = {"GrainWeight": "Grain weight (yield)", "Gsw": "Stomatal conductance (Gsw)",
         "PhiPS2": "Photosynthesis efficiency (ΦPSII)", "Fertilizer": "Fertilizer level"}


def plot(run):
    run = pathlib.Path(run)
    m = json.load(open(run / "metrics.json"))
    h = pd.read_csv(run / "history.csv")
    p = pd.read_csv(run / "predictions.csv")
    fig = plt.figure(figsize=(22, 9))
    # 1 training curves
    ax = fig.add_subplot(2, 4, 1)
    ax.plot(h.epoch, h.train_loss, label="train"); ax.plot(h.epoch, h.val_loss, label="validation")
    ax.axvline(m["best_epoch"], color="k", ls="--", lw=1, label=f"best epoch {m['best_epoch']}")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax.set_title("Training curves"); ax.legend()
    # 2 confusion matrix
    ax = fig.add_subplot(2, 4, 2)
    cm = np.zeros((4, 4), int)
    for t, q in zip(p.cultivar_true, p.cultivar_pred):
        cm[t, q] += 1
    ax.imshow(cm, cmap="Blues")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks(range(4), CULTIVARS, rotation=30); ax.set_yticks(range(4), CULTIVARS)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    c = m["cultivar"]
    ax.set_title(f"Cultivar: accuracy {c['accuracy']:.3f}, macro-F1 {c['macro_f1']:.3f}")
    # 3 per-class accuracy
    ax = fig.add_subplot(2, 4, 3)
    acc = cm.diagonal() / np.maximum(cm.sum(1), 1)
    ax.bar(CULTIVARS, acc, color="tab:green"); ax.set_ylim(0, 1); ax.set_title("Accuracy per cultivar")
    for i, v in enumerate(acc):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center")
    # 4 summary text
    ax = fig.add_subplot(2, 4, 4); ax.axis("off")
    lines = [f"Run: {run.name}", f"split: {m['split']}  seed {m['seed']}  fold {m['fold']}",
             f"backbone: {m['backbone']}", f"test leaves: {m['n_test']}", "",
             f"Cultivar accuracy: {c['accuracy']:.3f}", f"Cultivar macro-F1: {c['macro_f1']:.3f}"]
    lines += [f"{NAMES[t]}: R² {m[t]['R2']:.3f}, RMSE {m[t]['RMSE']:.4g}" for t in REGRESSION_TARGETS]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10)
    # 5-8 predicted vs true
    for k, t in enumerate(REGRESSION_TARGETS):
        ax = fig.add_subplot(2, 4, 5 + k)
        x, y = p[f"{t}_true"].round(4), p[f"{t}_pred"]
        if t == "Fertilizer":
            groups = sorted(x.unique())
            ax.boxplot([y[x == g] for g in groups], positions=range(len(groups)))
            ax.set_xticks(range(len(groups)), [f"{g:g}" for g in groups])
        else:
            ax.scatter(x, y, s=6, alpha=0.4)
            lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
            ax.plot([lo, hi], [lo, hi], "k--", lw=1)
        ax.set_xlabel("true"); ax.set_ylabel("predicted")
        ax.set_title(f"{NAMES[t]}: R² {m[t]['R2']:.3f}")
    fig.suptitle(f"Step 5 baseline results — {run.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(run / "figures.png", dpi=110)
    plt.close(fig)
    print("saved", run / "figures.png")


if __name__ == "__main__":
    for r in sys.argv[1:]:
        if not (pathlib.Path(r) / "metrics.json").exists():
            print("skipped (run not finished):", r)
            continue
        plot(r)
