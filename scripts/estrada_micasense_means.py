"""Mean real MicaSense value over leaf pixels (value > 0) for every leaf, stage and band.
Resumable: appends to results/step04_simulator_check/estrada_micasense_means.csv."""
import csv, pathlib, sys, time
import numpy as np, tifffile
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam.data import estrada
out = pathlib.Path("results/step04_simulator_check/estrada_micasense_means.csv")
tab, _ = estrada.load("data/raw/estrada_leaves/DATASET")
done = set()
if out.exists():
    done = {(r["crop"], int(r["leaf"]), int(r["stage"])) for r in csv.DictReader(open(out))}
new = not out.exists()
f = open(out, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["crop", "leaf", "stage"] + [f"b{b}_mean" for b in range(1, 6)] + [f"b{b}_pixels" for b in range(1, 6)])
t0 = time.time(); n = 0
for _, r in tab[tab.micasense_ok].iterrows():
    key = (r.crop, int(r.leaf), int(r.stage))
    if key in done: continue
    means, pix = [], []
    for p in r.micasense_files:
        a = tifffile.imread(p).astype(np.float64); m = a > 0
        means.append(a[m].mean() if m.any() else np.nan); pix.append(int(m.sum()))
    w.writerow(list(key) + [round(x, 2) for x in means] + pix); n += 1
    if time.time() - t0 > 150: break
f.close()
total = sum(1 for _ in open(out)) - 1
print(f"added {n}, total {total} of {int(tab.micasense_ok.sum())}")
