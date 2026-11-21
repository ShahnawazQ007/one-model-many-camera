"""Collect Step 9 results from saved Kaggle notebooks (kaggle_runs/06*_step9_cv.ipynb) or Kaggle log text files,
combine the folds, and write paper/data/cv_per_fold.csv + runs/cv_summary/*.

python scripts/cv_from_notebooks.py kaggle_runs/06a_step9_cv.ipynb kaggle_runs/06b_step9_cv.ipynb [more .ipynb/.txt]
"""
import io
import json
import pathlib
import re
import subprocess
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
frames = []
for f in sys.argv[1:]:
    p = pathlib.Path(f)
    if p.suffix == ".ipynb":
        nb = json.load(open(p))
        text = "\n".join("".join(o.get("text", "")) if isinstance(o.get("text", ""), list) else o.get("text", "")
                         for c in nb["cells"] if c["cell_type"] == "code" for o in c.get("outputs", []))
    else:
        text = p.read_text()
    found = 0
    # 1) full per-camera CSV printed by the summary cell
    m = re.search(r"=== CV_ALL_CAMERAS_CSV ===\n(.*)", text, re.S)
    if m:
        frames.append(pd.read_csv(io.StringIO(m.group(1).strip()))); found += 1
    # 2) per-run CSVs printed by run_cv.py (newer version) right after each run
    for name, csv in re.findall(r"RESULT_CSV (\S+)\n(.*?)END_CSV", text, re.S):
        mm = re.match(r"(baseline_eval|svct|interp)_s(\d+)_f(\d+)$", name)
        lab = {"baseline_eval": "204-band baseline", "svct": "SVCT (wavelength stem)", "interp": "SVCT (interp)"}[mm[1]]
        frames.append(pd.read_csv(io.StringIO(csv)).assign(method=lab, seed=int(mm[2]), fold=int(mm[3]))); found += 1
    print(f"{f}: {found} result block(s)")
if not frames:
    sys.exit("no results found")
df = pd.concat(frames, ignore_index=True).drop_duplicates(["method", "seed", "fold", "camera", "condition"])
out = ROOT / "runs/cv_merged"; out.mkdir(parents=True, exist_ok=True)
for (m, s, fo), g in df.groupby(["method", "seed", "fold"]):
    key = {"204-band baseline": "baseline_eval", "SVCT (wavelength stem)": "svct", "SVCT (interp)": "interp"}[m]
    d = out / f"{key}_s{s}_f{fo}"; d.mkdir(exist_ok=True)
    g.drop(columns=["method", "seed", "fold"]).to_csv(d / "camera_eval.csv", index=False)
subprocess.run([sys.executable, str(ROOT / "scripts/summarize_cv.py"), "--cv", str(out), "--out", str(ROOT / "runs/cv_summary")], check=True)
(ROOT / "paper/data").mkdir(parents=True, exist_ok=True)
pd.read_csv(ROOT / "runs/cv_summary/cv_per_fold.csv").to_csv(ROOT / "paper/data/cv_per_fold.csv", index=False)
print("wrote paper/data/cv_per_fold.csv -> now run: python paper/make_tables.py")
