# Paper draft (LaTeX)

- `main.tex` — paper draft (Elsevier `elsarticle` on Overleaf; falls back to `article` elsewhere)
- `results_report.tex` — step-by-step results report
- `tables/*.tex` and `figures/*.pdf` are generated from the result files — do not edit numbers by hand:
  `python paper/make_figures.py && python paper/make_tables.py`

## Adding the Step 9 cross-validation results
1. Save the finished Kaggle notebooks to `kaggle_runs/06a_step9_cv.ipynb` and `kaggle_runs/06b_step9_cv.ipynb`
2. `python scripts/cv_from_notebooks.py kaggle_runs/06a_step9_cv.ipynb kaggle_runs/06b_step9_cv.ipynb`
3. `python paper/make_tables.py`, then update the text of Section 4.5 and recompile

## Overleaf
Upload `anycam_paper_overleaf.zip` (New Project → Upload Project). Compiler: pdfLaTeX. Main document: `main.tex`.
