"""Step 9 / E5: plot-grouped cross-validation of the frozen Step 7 configurations.

For every fold (seed 0 unless --seed) three models are trained on the same split:
  baseline  - Step 5 Model A (204-band ResNet18, upscale 2, strong aug), then tested on all cameras (eval_cameras.py)
  svct      - SVCT final, wavelength-conditioned stem (frozen round-3 settings)
  interp    - SVCT final, bands interpolated back to 204 (no wavelength layer)
With 2 GPUs the work runs as two parallel streams: GPU0 = baseline + eval + svct, GPU1 = interp.
Finished runs are skipped, so the script can be re-run after an interruption.

python scripts/run_cv.py --data DATA --folds 0 1 2
"""
import argparse
import os
import pathlib
import subprocess
import sys
import threading
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="runs/cv")
p.add_argument("--workers", type=int, default=2)
p.add_argument("--smoke", action="store_true", help="tiny runs to check the pipeline")
a = p.parse_args()
out = pathlib.Path(a.out); (out / "logs").mkdir(parents=True, exist_ok=True)
PY = sys.executable
SVCT = ["--views", "1", "--p-full", "0.3", "--radio", "0.5", "--sampler", "mixed", "--family-weights", "0.45,0.35,0.2",
        "--select", "acc", "--epochs", "150", "--patience", "50"]
BASE = ["--upscale", "2", "--aug", "strong", "--epochs", "50", "--patience", "12"]
if a.smoke:
    SVCT = SVCT[:-4] + ["--epochs", "1", "--patience", "1", "--smoke"]
    BASE = BASE[:-4] + ["--smoke"]


def jobs(kind, fold):
    tag = f"s{a.seed}_f{fold}"
    split = ["--data", a.data, "--split", "cv", "--seed", str(a.seed), "--fold", str(fold)]
    if kind == "baseline":
        d = out / f"baseline_{tag}"
        return [(d / "metrics.json", [PY, "experiments/train_baseline.py", *split, *BASE, "--workers", str(a.workers),
                                       "--out", str(d)]),
                (out / f"baseline_eval_{tag}" / "camera_eval.csv",
                 [PY, "experiments/eval_cameras.py", "--data", a.data, "--run", str(d), "--workers", str(a.workers),
                  "--out", str(out / f"baseline_eval_{tag}")] + (["--limit", "16", "--drift-draws", "1"] if a.smoke else []))]
    stem = "hyper" if kind == "svct" else "interp"
    d = out / f"{kind}_{tag}"
    return [(d / "camera_eval.csv", [PY, "experiments/train_svct.py", *split, "--stem", stem, *SVCT,
                                     "--workers", str(a.workers), "--out", str(d)])]


def stream(gpu, kinds, status):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
    for fold in a.folds:
        for kind in kinds:
            for done_file, cmd in jobs(kind, fold):
                name = pathlib.Path(cmd[cmd.index("--out") + 1]).name
                if done_file.exists():
                    status[gpu] = f"{name}: already done"
                    continue
                log = open(out / "logs" / f"{name}.log", "w")
                t0 = time.time()
                status[gpu] = f"{name}: running"
                r = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                log.close()
                msg = "ok" if r.returncode == 0 and done_file.exists() else f"FAILED (exit {r.returncode}, see logs/{name}.log)"
                print(f"[GPU{gpu}] {name} {msg} in {(time.time() - t0) / 60:.0f} min", flush=True)
                if msg != "ok":
                    print("".join(open(out / "logs" / f"{name}.log").readlines()[-15:]), flush=True)


import torch  # noqa: E402
n_gpu = max(torch.cuda.device_count(), 1)
plan = [(0, ["baseline", "svct"]), (1, ["interp"])] if n_gpu >= 2 else [(0, ["baseline", "svct", "interp"])]
print(f"{n_gpu} GPU(s); folds {a.folds} seed {a.seed}; streams: {plan}", flush=True)
status = {g: "starting" for g, _ in plan}
threads = [threading.Thread(target=stream, args=(g, k, status)) for g, k in plan]
for t in threads:
    t.start()
t0 = time.time()
while any(t.is_alive() for t in threads):
    time.sleep(5 if a.smoke else 600)
    if any(t.is_alive() for t in threads):
        print(f"... {(time.time() - t0) / 60:.0f} min | " + " | ".join(f"GPU{g} {s}" for g, s in status.items()), flush=True)
print(f"all done in {(time.time() - t0) / 60:.0f} min", flush=True)
