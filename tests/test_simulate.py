import sys, pathlib
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam import (CameraProfile, list_cameras, load_camera, simulate, srf_matrix,
                    band_coverage, perturb_camera, apply_radiometric, sample_virtual_camera)
from anycam.simulate import shift_camera

WL = np.linspace(400, 1000, 204)  # Specim-IQ-like sampling


def test_registry_loads_all_cameras():
    names = list_cameras()
    assert len(names) >= 7
    for n in names:
        cam = load_camera(n)
        assert len(cam) >= 3 and all(b.fwhm > 0 for b in cam.bands)


def test_identity_camera_returns_input():
    """A camera with one very narrow band at each source wavelength reproduces the cube."""
    ident = CameraProfile.from_lists("identity", WL.tolist(), [1e-3] * len(WL))
    cube = np.random.default_rng(0).uniform(0, 1, size=(len(WL), 6, 5))  # (C,H,W)
    out = simulate(cube, WL, ident, axis=0)
    np.testing.assert_allclose(out, cube, atol=1e-9)


def test_rows_normalised_and_flat_spectrum_preserved():
    for n in list_cameras():
        cam = load_camera(n)
        M = srf_matrix(WL, cam)
        np.testing.assert_allclose(M.sum(1), 1.0, atol=1e-9)
        flat = np.full((3, len(WL)), 0.37)
        np.testing.assert_allclose(simulate(flat, WL, cam), 0.37, atol=1e-9)


def test_band_value_is_local_average():
    """A step spectrum: bands far below the step read ~0, far above read ~1."""
    spec = (WL > 700).astype(float)[None, :]
    cam = load_camera("micasense_rededge_mx")
    out = simulate(spec, WL, cam)[0]
    names = [b.name for b in cam.bands]
    assert out[names.index("blue")] < 1e-3 and out[names.index("nir")] > 0.999
    assert 0.5 < out[names.index("red_edge")] < 1.0  # straddles the step at 700 nm


def test_out_of_range_band_rejected():
    cam = CameraProfile.from_lists("bad", [1100.0], [20.0])
    with pytest.raises(ValueError):
        simulate(np.ones((1, len(WL))), WL, cam)
    assert band_coverage(WL, 700, 20) > 0.999


def test_axis_handling_images_and_batches():
    cam = load_camera("sentinel2a_vnir")
    rng = np.random.default_rng(1)
    chw = rng.uniform(0, 1, (len(WL), 4, 7))
    bchw = rng.uniform(0, 1, (2, len(WL), 4, 7))
    assert simulate(chw, WL, cam, axis=0).shape == (len(cam), 4, 7)
    assert simulate(bchw, WL, cam, axis=1).shape == (2, len(cam), 4, 7)


def test_perturbations_within_ranges():
    rng = np.random.default_rng(2)
    cam = load_camera("dji_p4_multispectral")
    for _ in range(50):
        p = perturb_camera(cam, rng, max_shift_nm=10, fwhm_scale=0.25)
        for a, b in zip(cam.bands, p.bands):
            assert abs(a.center - b.center) <= 10 + 1e-9
            assert 0.75 * a.fwhm - 1e-9 <= b.fwhm <= 1.25 * a.fwhm + 1e-9
    s = shift_camera(cam, 5)
    assert all(abs(b.center - a.center - 5) < 1e-9 for a, b in zip(cam.bands, s.bands))


def test_radiometric_noise_level():
    rng = np.random.default_rng(3)
    x = np.full((10000, 5), 0.5)
    y = apply_radiometric(x, rng, gain_range=(1, 1), offset_range=(0, 0), snr_db=(40, 40))
    snr = 20 * np.log10(np.sqrt(np.mean(y ** 2)) / np.std(y - 0.5))
    assert 39 < snr < 41


def test_virtual_cameras_valid_and_leave_camera_out():
    rng = np.random.default_rng(4)
    held = [load_camera("micasense_rededge_mx"), load_camera("parrot_sequoia")]
    red_edge_hits = 0
    for _ in range(500):
        v = sample_virtual_camera(rng, held_out=held)
        assert 3 <= len(v) <= 12
        assert all(10 <= w <= 60 for w in v.fwhms)
        srf_matrix(WL, v)  # every band inside the source range
        red_edge_hits += any(700 <= c <= 740 for c in v.centers)
        for h in held:
            if len(h) == len(v):
                ref = np.array(h.centers)
                assert not all(np.min(np.abs(ref - c)) <= 15 for c in v.centers)
    assert red_edge_hits > 200  # ~50% forced + chance
