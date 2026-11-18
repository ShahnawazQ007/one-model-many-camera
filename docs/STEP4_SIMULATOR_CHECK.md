# Step 4 — Is the camera simulator realistic? (18 Nov 2026)

## 4a. Estrada leaves: lab spectrum vs REAL MicaSense camera (1,594 leaf samples)
- Simulated the 5 MicaSense RedEdge-MX bands from each leaf's ASD spectrum and compared with the
  mean leaf value of the real MicaSense images of the same leaf and drying stage.
- Band identity: matching simulated to real bands recovered exactly MicaSense's documented order
  (file _1 blue, _2 green, _3 red, _4 NIR, _5 red-edge) without being told it.
- Brightness agreement is weak (cross-validated R2: red-edge 0.60, green 0.34, others < 0.1) because the
  released MicaSense files are uncalibrated camera counts with no exposure metadata (no panel, no EXIF).
- Conclusion: Estrada's MicaSense images cannot serve as an absolute simulator check. For E6 they need a
  per-band calibration (e.g., a few calibration leaves, like a reflectance panel in the field).
- Files: results/step04_simulator_check/estrada_*.csv, estrada_simulated_vs_real.png

## 4b. DeepHS fruit: two REAL calibrated hyperspectral cameras, same fruit side (220 pairs)
- Simulated the same low-cost cameras from Specim FX10 and from Corning microHSI 410 (no fitting).
- Agreement (Pearson r): green/red 0.998, red-edge 0.994, NIR 0.991, blue 0.94 (MicaSense);
  same pattern for Parrot Sequoia, DJI P4M, Sentinel-2 bands in 408-901 nm, and RGB.
- Main difference is a per-band brightness gain between the two real cameras:
  Corning = 1.35 x FX10 (blue), 1.00 (green), 0.98 (red), 0.97 (red-edge), 0.86 (NIR).
  After a gain/offset correction, R2 = 0.88 (blue), 0.98-0.997 (other bands).
- Spectral shape agreement per fruit side (410-900 nm): see deephs_fx10_vs_corning_simulated.csv.
- Decision: real camera-to-camera gains reach 0.86-1.35, wider than the planned +/-10%, so the simulator's
  default calibration gain range is now 0.8-1.25.
- Files: results/step04_simulator_check/deephs_*.csv, deephs_fx10_vs_corning_simulated.png

## Gate 1 (simulator part): PASSED
The simulator reproduces real-camera band behaviour (r 0.94-0.998 between two real cameras); the remaining
real-world difference is a per-band gain, which the training augmentation now covers.
