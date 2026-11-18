# Data report — checked 18 Nov 2026

Raw downloads: data/data/ (zips + rar). Unpacked: data/raw/estrada_leaves only.
HyperLeaf and DeepHS are read directly from their zip files (no 60 GB unpacking needed).

## 1. HyperLeaf2024 — data/data/HyperLeaf2024.zip  (READY)
- 2,410 images: images/NNNNN.tiff, shape (204 bands, 48, 352), uint16 on a 0-65,535 scale -> divide by 65,535
  (RELATIVE reflectance, not absolute). ~80-92% of each image is background, already 0; the leaf is ~10-20%
  (1,500-3,100 pixels). Median leaf spectrum looks right: green peak, red dip, red-edge rise, NIR ~0.8.
- No wavelength list in the files: Specim IQ, 204 bands, ~397-1004 nm (assumed until confirmed).
- Labels: train.csv (1,590) + solution.csv (820 test, with answers) = all 2,410 labelled.
  Columns: GrainWeight, Gsw (stomatal conductance), PhiPS2, Fertilizer (0 / 0.5 / 1.0), cultivar one-hot
  (Heerup 630, Kvium 620, Sheriff 600, Rembrandt 560).
- No plot ID column, but RECOVERED: 36 unique GrainWeight values = 36 plots
  (4 cultivars x 3 fertilizer levels x 3 repeats); 20-110 leaves per plot.
  The official train/test split never shares a plot -> plot-grouped CV is possible.

## 2. DeepHS Fruit — Kaki.zip, Mango.zip, Papaya.zip  (READY: labels via GitHub)
- Each fruit has BOTH cameras: VIS = Specim FX10 (224 bands), VIS_COR = Corning microHSI 410 (249 bands).
- ENVI .hdr + .bin, float32, BIP. Kaki 64x64 px crops; Papaya larger (e.g. 592x257).
- Recordings (.bin): Kaki 395 VIS / 398 VIS_COR; Mango 562 / 562; Papaya 274 / 274.
- Labels FOUND (18 Nov): the official annotations.zip link is dead, but the same labels and the camera
  wavelength lists are in the DeepHS code repo: github.com/cogsys-tuebingen/deephs_fruit -> dataset/*_v2.json.
  Get them with: python scripts/get_deephs_labels.py  (saves to data/raw/deephs_fruit/labels/)
- Each JSON has: records (file paths, fruit, camera, day, side), annotations (ripeness_state:
  unripe / ripe / near overripe / overripe, firmness, sugar_content for most, weights), cameras (wavelengths).
- Labelled recordings per camera (front + back sides counted separately):
  Kaki 56 train / 12 val / 12 test; Mango 56 / 12 / 12; Papaya 42 / 9 / 9 -- identical counts on VIS and VIS_COR.
  So ~40 kaki, ~40 mango, ~30 papaya fruits are labelled: small, so the fruit test is a case study.

## 3. Estrada leaves — DATASET.rar -> data/raw/estrada_leaves/DATASET/  (READY)
- Avocado 104 leaves, Olive 111, Vineyard (grape) 104 = 319 leaves, each at 5 drying stages = 1,595 samples.
- Spectrometer: Spectral_<crop>.mat, 5 arrays (fresh, stage1-3, dry), each (leaves, 2151) = 350-2500 nm at 1 nm, reflectance 0-1.
- Real MicaSense images: Multispectral Images/leafNNNdS_B.tif (S = stage 0-4, B = band 1-5), uint16, LZW
  (needs the imagecodecs package).
- Labels (leaves x 5 stages): chlorophyll, nitrogen, FMC (dry and fresh basis), weight.
- 1 damaged file in the download: Olive/Multispectral Images/leaf004d2_1.tif -> leave out olive leaf 004, stage 2.
- Open question: MicaSense tifs are raw counts (uint16); check Sample_code_avocado.mlx for how they convert to reflectance.

## Step 3 run on the real data — 18 Nov 2026
- HyperLeaf: 2,410 leaves, 36 plots (3 per treatment, 20-110 leaves per plot), 15 plot-grouped CV splits + official split; no leakage.
- DeepHS: 440 labelled recordings = 110 fruits (40 kaki, 40 mango, 30 papaya) x 2 cameras x 2 sides; 220 camera pairs;
  ripeness per fruit: 62 perfect, 25 overripe, 23 unripe; all have sugar. VIS 224 x 64 x 64 (kaki), VIS_COR 249 bands;
  values are reflectance-like (about -0.2 .. 0.7). Fruit-grouped CV; no leakage.
- Estrada: 1,595 samples = 319 leaves (111 olive, 104 avocado, 104 grape) x 5 stages; 1,594 with all MicaSense files;
  spectra 0.013 .. 1.155. Leaf-grouped CV; no leakage.
- Splits and label tables saved in data/splits/ (SUMMARY.json).
