# Setup: GitHub + Kaggle (one time, simple steps)

Code lives on **GitHub**. Data and training live on **Kaggle**. Kaggle fetches both.

## Part 1 — Put the code on GitHub (15 min)
1. Go to **github.com** → **Sign up** (skip if you have an account).
2. Go to **desktop.github.com** → Download for macOS → open it → drag **GitHub Desktop** into Applications.
3. Open GitHub Desktop → **Sign in to GitHub.com** → Authorize in the browser.
4. Menu **File → Add Local Repository… → Choose…** and pick:
   Documents → Advanced Computer Vision → CV_26 → Assignments → **anycam** → **Add Repository**.
   (If it says "not a Git repository", click **create a repository** → **Create Repository**.)
5. Bottom-left **Summary** box: type `first version` → click **Commit to main**.
6. Top: **Publish repository** → tick **Keep this code private** → **Publish Repository**.

Every time Claude changes the code: open GitHub Desktop → type a summary → **Commit** → **Push origin**.

## Part 2 — Get Kaggle ready (10 min)
1. Go to **kaggle.com** → sign in (or Register).
2. Click your picture (top right) → **Settings** → **Phone verification** → verify your phone.
   (This turns on Internet and GPU for notebooks.)
3. Open **kaggle.com/competitions/HyperLeaf2024** → **Rules** tab → **I Understand and Accept**.

## Part 3 — Let Kaggle read your private GitHub code (5 min)
1. On github.com: your picture → **Settings** → **Developer settings** (bottom left) →
   **Personal access tokens** → **Fine-grained tokens** → **Generate new token**.
2. Name: `kaggle`. Expiration: 90 days. **Repository access → Only select repositories → anycam**.
   **Permissions → Contents → Read-only**. Click **Generate token** and **copy** it.
   Never paste this token in a chat or anywhere public.
3. In each Kaggle notebook (Part 4): menu **Add-ons → Secrets → Add Secret**,
   Label `GITHUB_TOKEN`, Value = the token → Save, and tick it ON for that notebook.

## Part 4 — Run notebook 01 (download the extra data, once)
1. Kaggle → **Create → New Notebook** → menu **File → Import Notebook** →
   upload `kaggle/01_download_extra_data.ipynb` from the anycam folder.
2. Right panel: **Settings → Internet → On**.
3. Click **Run All**. Wait until it finishes.
4. Top right: **Save Version → Save & Run All (Commit)** → Save. Wait for it to finish.
5. Open that saved version → **Output** tab → **New Dataset** → name `anycam-extra-data` → **Private** → Create.

## Part 5 — Run notebook 02 (check data + first simulation)
1. **Create → New Notebook → File → Import Notebook** → upload `kaggle/02_setup_check_and_first_simulation.ipynb`.
2. In the first code cell, change `YOUR_GITHUB_USERNAME` to your GitHub username.
3. Right panel: **Add Input** → Competitions → **HyperLeaf2024**; **Add Input** → Your Work → Datasets → **anycam-extra-data**.
4. **Settings → Internet → On**. Add the `GITHUB_TOKEN` secret (Part 3, step 3).
5. **Run All**. At the end, download the files in `anycam/reports/` and send them to Claude.
