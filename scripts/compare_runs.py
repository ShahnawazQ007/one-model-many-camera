"""Table of all runs: python scripts/compare_runs.py runs/*  -> prints and saves runs_comparison.csv"""
import json, pathlib, sys
import pandas as pd
rows = []
for r in sys.argv[1:]:
    f = pathlib.Path(r) / "metrics.json"
    if not f.exists():
        continue
    m = json.load(open(f))
    rows.append({"run": pathlib.Path(r).name, "split": m["split"], "backbone": m["backbone"],
                 "upscale": m.get("upscale", 1.0), "aug": m.get("aug", "flip"), "best_epoch": m["best_epoch"],
                 "cultivar_acc": round(m["cultivar"]["accuracy"], 3), "cultivar_F1": round(m["cultivar"]["macro_f1"], 3),
                 **{f"{t}_R2": round(m[t]["R2"], 3) for t in ("GrainWeight", "Gsw", "PhiPS2", "Fertilizer")}})
df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("runs_comparison.csv", index=False)
