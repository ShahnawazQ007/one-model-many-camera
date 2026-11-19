import pathlib, sys
import numpy as np
import torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam.models import MultiTaskNet
from anycam.train_utils import BandNorm, classification_metrics, r2_rmse


def test_multitask_forward_204_bands():
    m = MultiTaskNet(204, pretrained=False)
    lc, lr = m(torch.randn(2, 204, 48, 352))
    assert lc.shape == (2, 4) and lr.shape == (2, 4)


def test_widened_conv_keeps_response_for_flat_input():
    """A flat input gives the same first-layer output whatever the band count (weights are spread)."""
    import timm
    ref = timm.create_model("resnet18", pretrained=False, num_classes=0)
    m = MultiTaskNet(204, pretrained=False)
    m.backbone.load_state_dict({k: v for k, v in ref.state_dict().items() if k != "conv1.weight"}, strict=False)
    with torch.no_grad():
        m.backbone.conv1.weight.copy_(ref.conv1.weight.mean(1, keepdim=True).repeat(1, 204, 1, 1) * 3 / 204)
        x3, x204 = torch.full((1, 3, 32, 32), 0.5), torch.full((1, 204, 32, 32), 0.5)
        torch.testing.assert_close(m.backbone.conv1(x204), ref.conv1(x3), rtol=1e-4, atol=1e-5)


def test_bandnorm_keeps_background_zero():
    rng = np.random.default_rng(0)
    c = rng.uniform(0.1, 0.9, (5, 4, 6)).astype(np.float32); c[:, 0, :] = 0
    n = BandNorm.fit([c])
    z = n(c)
    assert np.all(z[:, 0, :] == 0)
    leaf = z[:, 1:, :].reshape(5, -1)
    np.testing.assert_allclose(leaf.mean(1), 0, atol=1e-5)


def test_metrics():
    assert classification_metrics([0, 1, 2, 3], [0, 1, 2, 3], 4)["accuracy"] == 1.0
    assert abs(r2_rmse([1, 2, 3], [1, 2, 3])["R2"] - 1) < 1e-9


def test_upscale_and_strong_aug():
    m = MultiTaskNet(204, pretrained=False, upscale=2.0)
    lc, lr = m(torch.randn(1, 204, 48, 352))
    assert lc.shape == (1, 4)
    from anycam.train_utils import strong_aug
    rng = np.random.default_rng(0)
    c = np.zeros((5, 20, 30), np.float32); c[:, 5:15, 5:25] = 1.0
    out = strong_aug(c, rng)
    assert out.shape == c.shape and (out[:, (out != 0).any(0)].size > 0)
    assert ((out != 0).any(0)).sum() == ((c != 0).any(0)).sum()  # leaf area kept (shift, no growth)
