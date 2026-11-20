import numpy as np
import torch
import torch.nn.functional as F

from anycam.cameras import load_camera
from anycam.data.hyperleaf import WAVELENGTHS
from anycam.simulate import simulate, sample_virtual_camera
from anycam.svct import GPURenderer, SpectralStats, SVCTNet, camera_tensors, consistency_loss, full_camera


def _cubes(n=4, seed=0):
    rng = np.random.default_rng(seed)
    base = np.linspace(0.05, 0.5, 204)[:, None, None]
    out = []
    for _ in range(n):
        c = (base * rng.uniform(0.7, 1.3) + rng.normal(0, 0.02, (204, 10, 14))).clip(0.01).astype(np.float32)
        c[:, :3] = 0
        out.append(c)
    return out


def test_stats_for_camera_match_simulated_bands():
    cubes = _cubes()
    st = SpectralStats.fit(cubes)
    cam = load_camera("micasense_rededge_mx")
    from anycam.simulate import srf_matrix
    mean, std = st.for_matrix(srf_matrix(WAVELENGTHS, cam))
    px = np.concatenate([simulate(c, WAVELENGTHS, cam, axis=0)[:, c.sum(0) > 0] for c in cubes], 1)
    assert np.allclose(mean, px.mean(1), rtol=1e-4) and np.allclose(std, px.std(1), rtol=1e-3)


def test_renderer_shapes_mask_and_normalisation():
    cubes = _cubes()
    st = SpectralStats.fit(cubes)
    r = GPURenderer(st, WAVELENGTHS, "cpu")
    x = torch.tensor(np.stack(cubes))
    cam = load_camera("sentinel2a_vnir")
    y = r.render(x, cam)
    assert y.shape == (4, 10, 10, 14) and torch.all(y[:, :, :3] == 0)
    leaf = y[:, :, 3:].permute(1, 0, 2, 3).reshape(10, -1)
    assert torch.allclose(leaf.mean(1), torch.zeros(10), atol=1e-3)
    assert r.render(x, cam, to_full=True).shape == (4, 204, 10, 14)
    g = torch.Generator().manual_seed(0); a = r.render(x, cam, radiometric=1.0, gen=g)
    g = torch.Generator().manual_seed(0); b = r.render(x, cam, radiometric=1.0, gen=g)
    assert torch.equal(a, b) and not torch.equal(a, y)
    assert torch.allclose(r.render(x, full_camera(WAVELENGTHS))[:, :, 3:].mean(), torch.tensor(0.0), atol=1e-3)


def test_svctnet_any_number_of_bands_and_rgb_init():
    torch.manual_seed(0)
    net = SVCTNet(pretrained=False).eval()
    for n in (3, 5, 10, 204):
        x = torch.randn(2, n, 48, 64)
        lc, lr_ = net(x, torch.linspace(420, 950, n), torch.full((n,), 30.0))
        assert lc.shape == (2, 4) and lr_.shape == (2, 4)
    # a camera with bands exactly at the R, G, B wavelengths starts as the pretrained RGB convolution
    rgb = torch.randn(1, 3, 32, 32)
    w = net.stem.rgb
    ref = F.conv2d(rgb, w, stride=2, padding=3)
    got = net.stem(rgb, torch.tensor([610.0, 540.0, 460.0]), torch.tensor([70.0, 80.0, 80.0]))
    assert torch.allclose(ref, got, atol=1e-5)


def test_gradients_reach_hypernetwork_and_consistency_is_zero_for_same_output():
    net = SVCTNet(pretrained=False).train()
    x = torch.randn(4, 6, 48, 64)
    c, w = camera_tensors(sample_virtual_camera(np.random.default_rng(0), n_bands=(6, 6)), "cpu")
    out = net(x, c, w)
    (F.cross_entropy(out[0], torch.tensor([0, 1, 2, 3])) + out[1].pow(2).mean()).backward()
    assert net.stem.mlp[-1].weight.grad.abs().sum() > 0
    o = (torch.randn(3, 4), torch.randn(3, 4))
    assert consistency_loss(o, o).item() < 1e-6


def test_structured_sampler_respects_held_out_and_range():
    from anycam.cameras import list_cameras
    from anycam.simulate import _too_close
    from anycam.svct import sample_structured_camera, distill_loss
    held = [load_camera(n) for n in list_cameras()]
    rng = np.random.default_rng(0)
    kinds = set()
    for _ in range(300):
        c = sample_structured_camera(rng, held_out=held)
        assert 3 <= len(c) <= 12
        assert not any(_too_close(c.centers, h, 15.0) for h in held if len(h) == len(c))
        kinds.add(len(c))
        from anycam.simulate import srf_matrix
        srf_matrix(WAVELENGTHS, c)   # raises if a band falls outside the sensor range
    assert {3, 4, 5}.issubset(kinds)
    o = (torch.randn(3, 4), torch.randn(3, 4))
    assert distill_loss(o, o).item() < 1e-6
