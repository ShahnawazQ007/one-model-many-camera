"""Mean reflectance spectrum of the central part of each labelled fruit recording (both cameras).
Resumable: results/step04_simulator_check/deephs_mean_spectra.npz (+ a progress csv)."""
import csv, pathlib, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam.data import deephs
OUT = pathlib.Path("results/step04_simulator_check"); OUT.mkdir(parents=True, exist_ok=True)
prog = OUT / "deephs_mean_spectra.csv"
fr, wl = deephs.load_labels("data/raw/deephs_fruit/labels")
reader = deephs.DeepHSImages("data/data")
done = set(pd.read_csv(prog).data_file) if prog.exists() else set()
new = not prog.exists()
f = open(prog, "a", newline=""); w = csv.writer(f)
if new: w.writerow(["data_file", "camera", "fruit", "fruit_id", "side", "h", "w", "spectrum"])
order = fr.sort_values("fruit", key=lambda s: s.map({"Kaki": 0, "Mango": 1, "Papaya": 2}))
t0 = time.time(); n = 0
for _, r in order.iterrows():
    if r.data_file in done: continue
    c = reader(r)                                   # (bands, H, W)
    B, H, W = c.shape
    core = c[:, H // 4: 3 * H // 4, W // 4: 3 * W // 4].reshape(B, -1)   # central 50% x 50%
    spec = np.median(core, axis=1)
    w.writerow([r.data_file, r.camera, r.fruit, r.fruit_id, r.side, H, W, " ".join(f"{v:.5f}" for v in spec)])
    f.flush(); n += 1
    if time.time() - t0 > 150: break
f.close()
tot = pd.read_csv(prog)
print(f"added {n}; total {len(tot)} of {len(fr)}:", tot.groupby(['fruit', 'camera']).size().to_dict())
