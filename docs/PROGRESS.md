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
