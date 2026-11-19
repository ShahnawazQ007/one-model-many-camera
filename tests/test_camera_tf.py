import numpy as np

from anycam.cameras import load_camera
from anycam.camera_tf import SimulatedCamera, back_to_full_matrix
from anycam.data.hyperleaf import WAVELENGTHS


def test_back_to_full_rows_sum_to_one_and_hit_centres():
    cam = load_camera("micasense_rededge_mx")
    W = back_to_full_matrix(cam, WAVELENGTHS)
    assert W.shape == (204, 5)
    assert np.allclose(W.sum(1), 1)
    i = np.argmin(np.abs(WAVELENGTHS - cam.centers[2]))
    assert W[i, 2] > 0.9


def test_simulated_camera_shapes_mask_and_determinism():
    rng = np.random.default_rng(0)
    cube = rng.uniform(0.05, 0.6, (204, 8, 12)).astype(np.float32)
    cube[:, :2] = 0  # background
    cam = load_camera("sentinel2a_vnir")
    y = SimulatedCamera(cam, WAVELENGTHS)(cube)
    assert y.shape == (10, 8, 12) and np.all(y[:, :2] == 0) and np.all(y[:, 2:] > 0)
    full = SimulatedCamera(cam, WAVELENGTHS, to_full=True, drift_seed=1, radiometric=True)
    a, b = full(cube), full(cube)
    assert a.shape == (204, 8, 12) and np.array_equal(a, b) and np.all(a[:, :2] == 0)
    # a flat spectrum stays flat through simulate + back-to-full
    flat = np.full((204, 4, 4), 0.3, np.float32)
    assert np.allclose(SimulatedCamera(cam, WAVELENGTHS, to_full=True)(flat), 0.3, atol=1e-5)
