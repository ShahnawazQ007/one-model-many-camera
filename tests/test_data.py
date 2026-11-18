"""Step 3 tests: readers and leakage-free splits, on stand-in data with the real file formats."""
import io
import json
import pathlib
import sys
import zipfile

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam.data import deephs, estrada, hyperleaf, splits
from anycam.data.group_splits import grouped_cv

CULT = hyperleaf.CULTIVARS


def make_hyperleaf_zip(path, leaves_per_plot=6, H=6, W=8):
    import tifffile
    rng = np.random.default_rng(0)
    rows, pid, img = [], 0, 0
    for c in CULT:
        for f in (0.0, 0.5, 1.0):
            for rep in range(3):
                gw = 3000 + 100 * pid  # one grain weight per plot
                split = "test" if rep == 2 else "train"  # official split never shares a plot
                for _ in range(leaves_per_plot):
                    r = {"ImageId": img, "GrainWeight": gw, "Gsw": rng.uniform(.02, .1),
                         "PhiPS2": rng.uniform(.4, .7), "Fertilizer": f, **{k: int(k == c) for k in CULT}, "_s": split}
                    rows.append(r); img += 1
                pid += 1
    df = pd.DataFrame(rows)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("train.csv", df[df._s == "train"].drop(columns="_s").to_csv(index=False))
        te = df[df._s == "test"].drop(columns="_s")
        z.writestr("test.csv", te[["ImageId"]].to_csv(index=False))
        z.writestr("solution.csv", te.assign(Usage="Private").to_csv(index=False))
        for i in df.ImageId:
            b = io.BytesIO(); tifffile.imwrite(b, rng.integers(0, 6000, (204, H, W), dtype=np.uint16))
            z.writestr(f"images/{i:05d}.tiff", b.getvalue())
    return len(df)


@pytest.fixture(scope="module")
def hl(tmp_path_factory):
    p = tmp_path_factory.mktemp("hl") / "HyperLeaf2024.zip"
    n = make_hyperleaf_zip(p)
    return p, n


def test_hyperleaf_labels_and_plots(hl):
    p, n = hl
    df = hyperleaf.load_labels(p)
    assert len(df) == n and df.plot_id.nunique() == 36
    info = hyperleaf.check_plots(df)
    assert info["plots_in_both_official_splits"] == 0


def test_hyperleaf_image_from_zip(hl):
    p, _ = hl
    df = hyperleaf.load_labels(p)
    imgs = hyperleaf.HyperLeafImages(p)
    ds = hyperleaf.HyperLeafDataset(df, imgs, rows=range(5),
                                    target_stats=hyperleaf.target_stats(df, range(len(df))))
    item = ds[0]
    assert item["cube"].shape == (204, 6, 8) and item["cube"].dtype == np.float32
    assert 0 <= item["cube"].max() <= 0.6001 and item["targets"].shape == (4,)


def test_plot_cv_no_leakage_and_full_coverage(hl):
    p, _ = hl
    df = hyperleaf.load_labels(p)
    cv = splits.plot_grouped_cv(df)
    assert len(cv) == 15  # 5 folds x 3 seeds
    for s in cv:
        splits.assert_no_leakage(df, s)
        assert len(s["val_plots"]) == 3
        assert len(set(df.loc[s["val"], "Fertilizer"])) == 3  # one validation plot per fertilizer level
    for seed in (0, 1, 2):
        tested = sorted(p for s in cv if s["seed"] == seed for p in s["test_plots"])
        assert tested == list(range(36))  # every plot tested exactly once per seed
        sizes = [len(s["test_plots"]) for s in cv if s["seed"] == seed]
        assert min(sizes) >= 6 and max(sizes) <= 9 and sum(sizes) == 36
    off = splits.official_split(df)
    splits.assert_no_leakage(df, off)


def test_leakage_is_detected(hl):
    p, _ = hl
    df = hyperleaf.load_labels(p)
    bad = {"train": [0, 1], "val": [], "test": [2]}  # rows 0-2 are in the same plot
    with pytest.raises(AssertionError):
        splits.assert_no_leakage(df, bad)


def test_envi_roundtrip():
    cube = np.random.default_rng(1).random((5, 3, 4)).astype(np.float32)  # (bands, lines, samples)
    data = cube.transpose(1, 2, 0).tobytes()  # BIP
    hdr = "ENVI\nsamples = 4\nlines = 3\nbands = 5\nheader offset = 0\ndata type = 4\ninterleave = bip\nbyte order = 0\n"
    np.testing.assert_array_equal(deephs.envi_from_bytes(hdr, data), cube)


DEEPHS_LABELS = pathlib.Path("/home/claude/cogsys-tuebingen/deephs_fruit/dataset")


@pytest.mark.skipif(not DEEPHS_LABELS.exists(), reason="real DeepHS label files not available")
def test_deephs_real_labels_and_fruit_split():
    df, wl = deephs.load_labels(DEEPHS_LABELS)
    assert df.fruit_id.nunique() == 110
    assert (df.groupby("fruit_id").size() == 4).all()
    assert len(wl["VIS"]) == 224 and len(wl["VIS_COR"]) == 249
    pairs = deephs.camera_pairs(df)
    assert len(pairs) == 220  # 110 fruits x 2 sides
    cv = grouped_cv(df, "fruit_id", "ripeness")
    for s in cv:
        splits.assert_no_leakage(df, s, group_col="fruit_id")


def test_estrada_reader(tmp_path):
    from scipy.io import savemat
    root = tmp_path / "DATASET"
    rng = np.random.default_rng(2)
    for crop, low in estrada.CROPS.items():
        d = root / crop; (d / "Multispectral Images").mkdir(parents=True)
        n = 4
        savemat(d / f"Spectral_{low}.mat", {f"{s}_leaves_{low}": rng.random((n, 2151)) for s in estrada.STAGES})
        savemat(d / f"Chlorophyll_{low}.mat", {f"chlorophyll_{low}": rng.random((n, 5))})
        savemat(d / f"Nitrogen_{low}.mat", {f"nitrogen_{low}": rng.random((n, 5))})
        savemat(d / f"FMC_{low}.mat", {f"FMC_d_{low}": rng.random((n, 5)), f"FMC_f_{low}": rng.random((n, 5))})
        savemat(d / f"Weight_{low}.mat", {f"weight_{low}": rng.random((n, 5))})
        for leaf in range(n):
            for s in range(5):
                for b in range(1, 6):
                    (d / "Multispectral Images" / f"leaf{leaf:03d}d{s}_{b}.tif").write_bytes(b"x")
    table, spectra = estrada.load(root)
    assert len(table) == 3 * 4 * 5 and spectra.shape == (60, 2151)
    assert table.micasense_ok.all()
    cv = grouped_cv(table, "leaf_id", "crop", n_folds=2, seeds=(0,))
    for s in cv:
        splits.assert_no_leakage(table, s, group_col="leaf_id")
