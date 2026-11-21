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

## 19 Nov 2026 — Step 5 first result (Kaggle GPU, official split, ResNet18, 204 bands)
- Cultivar accuracy 0.730 (macro-F1 0.705); R2: grain weight 0.712, Gsw 0.114, PhiPS2 0.407, fertilizer 0.774.
- Best epoch 7, early stop at 15; train loss 0.33 vs validation 1.4-1.6 -> overfitting.
- Paper reference (ConvNeXt-S): cultivar 0.893, fertilizer R2 0.84, Gsw R2 0.19 (their split differs).
- Next: input upscaling (2x) + stronger augmentation; ConvNeXt-Tiny (kaggle/03b_step5_improved_baseline.ipynb).
- Re-run reproduced 5a exactly (seeded). CV seed0 fold0 (370 test leaves, unseen plots): cultivar 0.592 / F1 0.578;
  R2 grain 0.484, Gsw -0.05, PhiPS2 0.484, fertilizer 0.806. ~30 s/epoch on T4 at upscale 1.
- Confusion: Rembrandt <-> Heerup most confused. Gsw not predictable (paper 0.19 too). Fertilizer is the most robust target.
- Figures: results/step05_baseline/ (extracted from kaggle_runs/03_step5_baseline_training.ipynb).

## 20 Nov 2026 — Step 5 improved baseline (new_baseline_kaggle.ipynb)
| run | cultivar acc | F1 | grain R2 | Gsw R2 | PhiPS2 R2 | fert R2 |
|---|---|---|---|---|---|---|
| ResNet18 plain | 0.730 | 0.705 | 0.712 | 0.114 | 0.407 | 0.774 |
| A ResNet18 up2 strong | **0.821** | **0.805** | 0.584 | 0.077 | **0.524** | 0.759 |
| B ConvNeXt-T up2 strong | 0.768 | 0.764 | **0.714** | 0.037 | 0.471 | **0.785** |
- Decision: **A (ResNet18, upscale 2, strong aug) is the reference baseline** for Steps 6-7 (best cultivar, 3x fewer params, ~70 s/epoch on T4). ConvNeXt validation loss very unstable.
- Still overfits (train 0.37 vs val 1.2-1.5). Paper (ConvNeXt-S, own split) cultivar 0.893. Final numbers will come from plot-grouped CV.
- Figures + CSV: results/step05_baseline/.

## 20 Nov 2026 — Step 6 prepared (E1 + E2)
- `anycam/camera_tf.py`: SimulatedCamera (simulate, optional drift / gain-offset-noise, optional linear map back to 204 bands).
- `train_baseline.py --camera X`: camera-specific model ("one model per camera" reference).
- `experiments/eval_cameras.py`: 204-band model on 7 cameras x {nominal, drift (3 units), radiometric, drift+radio} -> camera_eval.csv/png.
- Kaggle: `kaggle/04_step6_camera_test.ipynb` (~2 h).

## 20 Nov 2026 — Step 6 RESULT (kaggle_runs/04_step6_camera_test.ipynb; results/step06_camera_test/)
204-band model (cultivar 0.821, fert R2 0.759) tested on simulated cameras (bands mapped back to 204 by interpolation):
- cultivar accuracy falls to 0.36 (phone RGB) - 0.61 (Sentinel-2); fertilizer R2 negative for 4/7 cameras (Parrot, MicaSense-no-RE ~ -1.2 to -1.4).
- drift (+-10 nm, +-25 % FWHM) changes little compared with the camera gap itself; gain/offset/noise costs the 204 model 0.82 -> 0.77 even on its own data.
- camera-specific models recover almost everything: MicaSense 0.760 / fert 0.778, Sentinel-2 0.718 / 0.791, phone RGB 0.795 / 0.720.
=> the information IS in a few bands; the failure is the model, not the camera. Target for SVCT: one model near the green bars on every camera.

## 20 Nov 2026 — Step 7 prepared (SVCT)
- `anycam/svct.py`: SpectralStats (any camera's normalisation from the 204-band mean/covariance), GPURenderer
  (SRF + per-sample gain/offset/noise on GPU), WavelengthStem (hyper-network filters per band, initialised from the
  ImageNet RGB filters interpolated over wavelength), SVCTNet, consistency loss.
- `experiments/train_svct.py`: random virtual cameras (3-12 bands, FWHM 10-100 nm, p_full 0.15), 7 real profiles held out
  (15 nm rule); validation on 6 fixed virtual cameras; test on 204 + 7 real x 4 conditions + 20 random cameras.
- Kaggle `kaggle/05_step7_svct.ipynb`: SVCT main, ablation interp-stem, ablation no-consistency; `scripts/plot_step7.py`.

## 20 Nov 2026 — Step 7 round 1 RESULT (kaggle_runs/05_step7_svct.ipynb; results/step07_svct_round1/)
Mean over the 7 held-out real camera profiles, ideal cameras (cultivar acc / fertilizer R2):
- 204-band model 0.522 / -0.364 | SVCT main 0.502 / +0.480 | no wavelength layer 0.556 / +0.377 | no consistency 0.639 / +0.456
- one model per camera (3 cameras only) 0.758 / 0.763. On 204 bands: baseline 0.821, SVCT variants 0.56-0.71.
- Regression fixed (negative -> positive R2 everywhere, robust to drift/noise); cultivar NOT yet; phone RGB worst (0.44-0.49).
Diagnosis: consistency weight 1.0 too strong (no-consistency variant is best); under-training (train loss 0.8-0.9 at
early stop, epoch 22; noisy val); virtual sampler rarely produces visible-only wide-band cameras; p_full 0.15 too low.
- Round 2 prepared (kaggle/05b_step7_round2.ipynb): teacher-mode distillation (full 204 view teaches the camera view,
  weight 0.5), structured camera sampler (rgb-like / multispectral / random, held-out rule kept), 80 epochs patience 30,
  checkpoint by val accuracy + R2; plus camera-specific models for the remaining 4 cameras.

## 21 Nov 2026 — Step 7 round 2 RESULT (kaggle_runs/05b_step7_round2.ipynb; results/step07_svct_round2/)
Mean over 7 held-out real cameras, ideal (cultivar / fert R2): 204-band 0.522 / -0.364 | SVCT r2 teacher 0.551 / 0.526 |
**no-consistency r2 0.664 / 0.527** | interp-stem r2 (teacher) 0.647 / 0.418 | one model per camera (all 7 now) 0.757 / 0.765.
- Best single model (no consistency): MicaSense 0.757 (= its own oracle 0.760), DJI 0.707 (oracle 0.782), Sentinel 0.677 (0.718);
  weakest phone RGB 0.510 (oracle 0.795). Robust: drift 0.664, drift+radio 0.639.
- Consistency loss hurts cultivar in both rounds (sym and teacher) -> drop from the method, report as a negative ablation.
- No-consistency run hit the 80-epoch limit while still improving (val acc 0.59 rising) -> train longer.
- Clean stem ablation still missing (interp WITHOUT consistency).
- Round 3 prepared (kaggle/05c_step7_round3.ipynb): final config = no consistency, p_full 0.3, radio 0.5, mixed sampler
  weights 0.45/0.35/0.2 (rgb-like/multispectral/random), 150 epochs patience 50; clean stem ablation (interp, same config).
  Method is frozen after this run.

## 21 Nov 2026 — Step 7 round 3 RESULT, METHOD FROZEN (kaggle_runs/05c_step7_round3.ipynb; results/step07_svct_round3/)
Mean over 7 held-out real cameras, ideal (cultivar / fert R2 / grain R2):
- 204-band model 0.522 / -0.364 / -0.685 | one model per camera 0.757 / 0.765 / 0.694
- SVCT + wavelength stem 0.670 / **0.582 / 0.528** (best epoch 21; robust: drift+radio 0.640 / 0.462)
- SVCT + interpolation (plain 204-band ResNet) **0.732** / 0.468 / 0.429 (phone RGB 0.768 vs oracle 0.795)
=> SVCT closes ~89 % of the cultivar gap (interp) and ~84 % of the fertilizer gap (stem). Stem = better regression,
   interp = better classification. Single split: confirm with CV/seeds before claiming which variant is better.

## Step 8 / E4 — REAL second camera (DeepHS fruit: train FX10 → test real Corning) — done
Fruit-grouped 5-fold CV × 3 seeds, 220 paired recordings (110 fruits). Mean over 15 folds:

| Test on REAL Corning | fruit acc | ripeness acc | sugar R² | log firmness R² |
|---|---|---|---|---|
| full (plain, FX10 only) | 0.838 | 0.558 | 0.547 | −0.364 |
| **SVCT (interp)** | **0.971** | 0.576 | **0.794** | **0.463** |
| SVCT (tokens) | 0.961 | 0.589 | 0.731 | 0.455 |
| Corning oracle (trained on Corning labels) | 0.995 | 0.580 | 0.793 | 0.566 |

- SVCT matches the "one model per camera" oracle on sugar (0.794 vs 0.793) without ever seeing a Corning label.
- Real Corning reduced to 7 simulated profiles: full sugar R² 0.01 vs SVCT 0.65; fruit 0.53 vs 0.93.
- No loss on the training camera (FX10 sugar 0.77 → 0.80).
- Ripeness is hard for every model (~0.58, including the oracle) → a data limit, not a camera problem.

## Step 8 / E6 — Estrada leaves (train ASD spectrometer → test real MicaSense + simulated cameras) — done
Leaf-grouped 5-fold CV × 3 seeds (1595 spectra, 319 leaves). Mean over 15 folds:

| Test | model | crop acc | stage acc | chlorophyll R² | water R² |
|---|---|---|---|---|---|
| 7 simulated cameras (mean) | full | 0.522 | 0.176 | −0.366 | −0.633 |
| | **svct_tok** | **0.843** | **0.539** | **0.733** | **0.564** |
| REAL MicaSense | full | 0.457 | 0.198 | −0.002 | −0.154 |
| | svct / svct_tok | 0.524 / 0.543 | 0.275 / 0.271 | 0.465 / 0.425 | 0.149 / 0.192 |
| | sim_ms (trained on simulated MicaSense) | 0.593 | 0.270 | 0.356 | −0.115 |
| | real_ms (trained on real MicaSense labels) | 0.927 | 0.525 | 0.800 | 0.514 |
| Full spectrum (same sensor) | full / svct | 0.980 / 0.924 | 0.899 / 0.681 | 0.860 / 0.812 | 0.934 / 0.750 |

- Simulated cameras: SVCT turns negative R² into 0.73 (chlorophyll) / 0.56 (water).
- Real MicaSense: SVCT beats the plain model on every task (chlorophyll −0.00 → 0.47), but a large gap to real_ms remains.
  The gap is a sim-to-real / calibration problem, not specific to SVCT: even a model trained on *simulated MicaSense* only reaches crop 0.59,
  and real vs simulated band agreement is low (R² blue 0.04, green 0.34, red 0.08, RE 0.61, NIR 0.01; uncalibrated DNs).
- Trade-off: on the original full spectrum SVCT costs accuracy (stage 0.90 → 0.68, water 0.93 → 0.75) — unlike the fruit data. To report honestly.

## Step 9 / E5 — cross-validation of the frozen Step 7 settings (Kaggle) — ready to run
- `scripts/run_cv.py` trains per fold: 204-band baseline (+ eval_cameras), SVCT wavelength stem, SVCT interp (two GPUs in parallel; finished runs are skipped).
- `scripts/summarize_cv.py` → mean (std over folds) of the mean over the 7 held-out cameras, per-fold table, wins vs baseline.
- Notebooks: `kaggle/06a_step9_cv.ipynb` (folds 0–2), `kaggle/06b_step9_cv.ipynb` (folds 3–4); seed 0. Camera-specific oracles are not repeated per fold (too costly) — the Step 7 single-split oracles remain the reference.
- Pipeline smoke-tested on a synthetic dataset (CPU).

## 21 Nov 2026 — Step 9 partial: folds 3–4 done (kaggle_runs/06b_step9_cv.ipynb)
Mean over 7 held-out cameras, ideal (cultivar / fert R2), mean (sd) over 2 folds:
- 204-band 0.332 (0.048) / −0.229 (0.285) | SVCT stem 0.758 (0.057) / 0.511 (0.168) | SVCT interp 0.748 (0.058) / 0.542 (0.094)
- Both SVCT variants beat the baseline in 2/2 folds on both metrics; drift+radio 0.72–0.73 accuracy. Phone RGB 0.171 → 0.68–0.75.
- Full 204 bands: baseline 0.781 / 0.812; stem 0.780 / 0.382 (unstable regression); interp 0.688 / 0.646.
- Folds 0–2 (06a) still running.
