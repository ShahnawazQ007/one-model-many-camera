# Step 1 — Download the data (your part)

Put every dataset under `data/raw/<name>/` inside this project. Raw data is never committed to git.

## 1. HyperLeaf2024 (main dataset) — Kaggle login needed
1. Sign in at kaggle.com and open https://www.kaggle.com/competitions/HyperLeaf2024
2. Read the **Rules** tab and click **I understand and accept** (needed before download).
   Note the data licence written there.
3. Download, either from the **Data** tab ("Download all"), or with the Kaggle API:
   ```
   pip install kaggle
   # put kaggle.json (Account > Create New Token) in ~/.kaggle/
   kaggle competitions download -c HyperLeaf2024 -p data/raw/hyperleaf2024
   cd data/raw/hyperleaf2024 && unzip *.zip
   ```
4. Run: `python scripts/inspect_data.py data/raw/hyperleaf2024`

## 2. DeepHS Fruit (real cross-camera test) — direct links, no login
Do NOT run the repo's full download script: it fetches the whole benchmark (~240 GB).
Download only these files into `data/raw/deephs_fruit/` and unzip each into its own folder:
- https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Papaya.zip   (FX10 + Corning)
- https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Kaki.zip     (FX10 + Corning)
- https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/Mango.zip    (FX10 + Corning)
- https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/annotations.zip
- https://cogsys.cs.uni-tuebingen.de/webprojects/DeepHS-Fruit-2023-Datasets/readme.txt
Optional (FX10 + NIR only, no Corning): Avocado.zip, Kiwi.zip at the same base path.
Camera folders: VIS = Specim FX10, VIS_COR = Corning microHSI 410, NIR = INNO-SPEC RedEye.
Then run: `python scripts/inspect_data.py data/raw/deephs_fruit`

## 3. Avocado / olive / grape leaves (third crop + real MicaSense camera, E6)
1. Paper: https://www.nature.com/articles/s41598-025-85714-8
2. Data: https://doi.org/10.6084/m9.figshare.26950660.v2 (Figshare, 'Download all') -> `data/raw/estrada_leaves/`
3. Note the licence shown on the Figshare page.

## 4. WHU-Hi (field scale) — academic use only
1. Terms and links: https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm
2. Fill in the short questionnaire, download the MATLAB or TIFF version to `data/raw/whu_hi/`.

## 5. Phone camera sensitivities (E5)
1. Paper: https://www.mdpi.com/1424-8220/21/15/4985 (20 phone cameras)
2. Data page: https://ohlab.kic.ac.jp/index/mobilespec_database (20 phones + average; txt or Excel per phone) -> `data/raw/phone_ssf/`

When 1 and 2 are done, send me the two DATA_REPORT.md files. They decide the split design.
