"""Download the DeepHS Fruit labels (ripeness, firmness, sugar) and camera wavelengths.

The official annotations.zip link is no longer reachable; the same labels are published
as JSON files in the DeepHS code repository (github.com/cogsys-tuebingen/deephs_fruit,
dataset/*_v2.json). This script saves them to data/raw/deephs_fruit/labels/.

Usage (from the project folder):  python scripts/get_deephs_labels.py
"""
import pathlib
import urllib.request

BASE = "https://raw.githubusercontent.com/cogsys-tuebingen/deephs_fruit/master/dataset/"
FILES = ["train_only_labeled_v2.json", "val_v2.json", "test_v2.json", "train_all_v2.json"]
out = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "deephs_fruit" / "labels"
out.mkdir(parents=True, exist_ok=True)
for f in FILES:
    print("downloading", f)
    urllib.request.urlretrieve(BASE + f, out / f)
print("saved to", out)
