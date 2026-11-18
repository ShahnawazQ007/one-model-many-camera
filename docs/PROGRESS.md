# Progress log

## 12 Oct 2026 — Step 1 (P0) started
- Project created; camera registry with 7 profiles (MicaSense RedEdge-MX, no-red-edge control,
  Parrot Sequoia, DJI P4 Multispectral, Sentinel-2A VNIR, generic RGB, generic RGB+NIR).
- Camera simulator + calibration perturbations + virtual-camera sampler written.
- 9 tests pass (identity camera, normalisation, step spectrum, range check, axes,
  perturbation ranges, noise SNR, virtual cameras + leave-camera-out).
- Assumptions to revisit: DJI "+/- 16 nm" read as half-FWHM; MicaSense/Sequoia widths read as FWHM;
  Gaussian responses everywhere; RGB profiles are placeholders until phone sensitivity data is in.
- Next: download data (docs/DATA_DOWNLOAD.md), run scripts/inspect_data.py, then data loaders + grouped splits.

## 17 Nov 2026 — Kaggle + GitHub setup prepared
- Plan: code on GitHub, data and training on Kaggle (GPU PC on hold).
- Added anycam/io.py (load cubes from .npy/.npz/.tif/.hdr, finds the band axis).
- Added scripts/first_simulation.py: real image -> simulated phone and MicaSense images + spectrum plot.
- inspect_data.py can now write its report elsewhere (Kaggle input folders are read-only).
- Added kaggle/01_download_extra_data.ipynb (DeepHS papaya/kaki/mango + Estrada) and
  kaggle/02_setup_check_and_first_simulation.ipynb (tests, GPU check, data reports, first simulation).
- Added docs/SETUP_GITHUB_KAGGLE.md (step-by-step).
- Not done yet: simulator check against real data (Step 4) — needs the data reports first.

## 18 Nov 2026 — Step 3: data readers and leakage-free splits
- anycam/data/hyperleaf.py: reads images straight from HyperLeaf2024.zip; labels from train.csv + solution.csv;
  36 plots recovered from GrainWeight. Assumed: Specim IQ wavelengths 397-1004 nm, reflectance = value / 10,000.
- anycam/data/splits.py: plot-grouped 5-fold CV (stratified by cultivar) x 3 seeds, 3 validation plots
  (one per fertilizer level) inside each training part; official competition split kept as an extra check.
- anycam/data/deephs.py: labels + wavelengths from the deephs_fruit repo JSON; ENVI read from the zips;
  110 labelled fruits (40 kaki, 40 mango, 30 papaya) x 4 recordings (2 cameras x 2 sides).
  The repo's own train/val/test lists share fruits between lists (leakage) -> we split by fruit.
- anycam/data/estrada.py: 319 leaves x 5 stages; ASD spectra 350-2500 nm; MicaSense file paths; labels.
- scripts/make_splits.py writes data/splits/*.json + SUMMARY.json. 16 tests pass (tests/test_data.py).

## 18 Nov 2026 — Step 4: simulator checked against real cameras (Gate 1, simulator part: PASSED)
- See docs/STEP4_SIMULATOR_CHECK.md. DeepHS FX10 vs Corning: r 0.94-0.998 per simulated band.
- Estrada MicaSense files are uncalibrated counts -> band order confirmed, absolute check not possible.
- Calibration gain range widened to 0.8-1.25 based on the measured real camera gains.

## 18 Nov 2026 — Step 5 prepared: baseline training
- anycam/models.py: ImageNet CNN (timm) with the first conv widened to 204 bands (pretrained RGB filters
  averaged and spread over bands) + cultivar head + 4 regression heads (multi-task).
- anycam/train_utils.py: per-band normalisation from training leaf pixels (background kept 0), flips, metrics.
- experiments/train_baseline.py: AdamW 3e-4, wd 0.05, cosine, 40 epochs, early stop on validation plots, AMP on GPU;
  saves metrics.json, predictions.csv, history.csv, model.pt. Smoke-tested on CPU; data path checked on the real
  HyperLeaf zip (29 ms per leaf).
- kaggle/03_step5_baseline_training.ipynb: runs it on a Kaggle GPU (official split + first CV fold).
