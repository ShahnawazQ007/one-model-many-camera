# Data report — checked 18 Nov 2026

Raw downloads: data/data/ (zips + rar). Unpacked: data/raw/estrada_leaves only.
HyperLeaf and DeepHS are read directly from their zip files (no 60 GB unpacking needed).

## 1. HyperLeaf2024 — data/data/HyperLeaf2024.zip  (READY)
- 2,410 images: images/NNNNN.tiff, shape (204 bands, 48, 352), uint16 (0..65k; looks like reflectance x 10,000).
- No wavelength list in the files: Specim IQ, 204 bands, ~397-1004 nm (assumed until confirmed).
- Labels: train.csv (1,590) + solution.csv (820 test, with answers) = all 2,410 labelled.
  Columns: GrainWeight, Gsw (stomatal conductance), PhiPS2, Fertilizer (0 / 0.5 / 1.0), cultivar one-hot
  (Heerup 630, Kvium 620, Sheriff 600, Rembrandt 560).
- No plot ID column, but RECOVERED: 36 unique GrainWeight values = 36 plots
  (4 cultivars x 3 fertilizer levels x 3 repeats); 20-110 leaves per plot.
  The official train/test split never shares a plot -> plot-grouped CV is possible.

## 2. DeepHS Fruit — Kaki.zip, Mango.zip, Papaya.zip  (IMAGES READY, LABELS MISSING)
- Each fruit has BOTH cameras: VIS = Specim FX10 (224 bands), VIS_COR = Corning microHSI 410 (249 bands).
- ENVI .hdr + .bin, float32, BIP. Kaki 64x64 px crops; Papaya larger (e.g. 592x257).
- Recordings (.bin): Kaki 395 VIS / 398 VIS_COR; Mango 562 / 562; Papaya 274 / 274.
- MISSING: annotations.zip (ripeness / firmness / sugar labels) and readme.txt. Download by hand:
  https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations.zip
  https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/readme.txt
  -> put both in data/data/
- Wavelength lists are not in the .hdr files: take them from the hsi_benchmark code.

## 3. Estrada leaves — DATASET.rar -> data/raw/estrada_leaves/DATASET/  (READY)
- Avocado 104 leaves, Olive 111, Vineyard (grape) 104 = 319 leaves, each at 5 drying stages = 1,595 samples.
- Spectrometer: Spectral_<crop>.mat, 5 arrays (fresh, stage1-3, dry), each (leaves, 2151) = 350-2500 nm at 1 nm, reflectance 0-1.
- Real MicaSense images: Multispectral Images/leafNNNdS_B.tif (S = stage 0-4, B = band 1-5), uint16, LZW
  (needs the imagecodecs package).
- Labels (leaves x 5 stages): chlorophyll, nitrogen, FMC (dry and fresh basis), weight.
- 1 damaged file in the download: Olive/Multispectral Images/leaf004d2_1.tif -> leave out olive leaf 004, stage 2.
- Open question: MicaSense tifs are raw counts (uint16); check Sample_code_avocado.mlx for how they convert to reflectance.
