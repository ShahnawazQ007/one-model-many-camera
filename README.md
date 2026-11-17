# AnyCam — One Model, Any Camera

Camera-agnostic plant-trait prediction: train once on hyperspectral images, use with
drone, satellite and phone cameras without retraining (SVCT: Stochastic Virtual Camera Training).

Plan, literature review and decisions: see the "One Model, Any Camera" planning document.

## Layout
```
anycam/            core library (cameras, simulator, sampler, stem, models, advisor)
  cameras/         one YAML per real camera (the camera registry)
data/              datasets (not in git) + loaders and grouped splits
baselines/         naive, oracle, VI+RF, band selection, dropout, STARS-style, Hyve, CARL, HRNET
experiments/       configs (E1-E6), run.py, analysis
app/               demo app + API
scripts/           data checks and utilities
tests/             simulator, leakage and API tests
docs/              download guide, progress log
```

## Quick start
```
pip install -r requirements.txt
python -m pytest -q
python scripts/list_cameras.py
```
