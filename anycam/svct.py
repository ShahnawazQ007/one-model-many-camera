"""SVCT: Stochastic Virtual Camera Training (Step 7).

Pieces
  SpectralStats      mean and covariance of TRAINING leaf pixels on the full 204-band grid. Any camera's
                     band statistics follow exactly: mean_b = M mu, var_b = diag(M Sigma M^T), so every
                     (virtual or real) camera is normalised consistently without refitting.
  GPURenderer        renders a batch of raw 204-band cubes as camera images on the GPU: spectral response
                     (M), optional calibration error (per-sample gain, offset, sensor noise), normalisation,
                     and optionally a linear map back to 204 bands (for the "interp" baseline).
  WavelengthStem     first convolution whose filter for each band is generated from that band's centre and
                     width (a small hyper-network). Initialised from the ImageNet RGB filters interpolated
                     over wavelength, so a 3-band RGB camera starts out as the pretrained network.
  SVCTNet            WavelengthStem + ResNet body + cultivar / regression heads; accepts any number of bands.
  consistency_loss   agreement between predictions of the same leaves seen through two different cameras.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_tf import back_to_full_matrix
from .cameras import CameraProfile
from .simulate import srf_matrix

RGB_CENTRES = (610.0, 540.0, 460.0)   # wavelengths assigned to the pretrained R, G, B filters (ImageNet order RGB)


# ---------------------------------------------------------------- statistics
class SpectralStats:
    def __init__(self, mean: np.ndarray, cov: np.ndarray):
        self.mean = np.asarray(mean, np.float64)
        self.cov = np.asarray(cov, np.float64)

    @classmethod
    def fit(cls, cubes):
        s, ss, n = 0.0, 0.0, 0
        for c in cubes:
            px = c[:, c.sum(0) > 0].astype(np.float64)        # (C, P) leaf pixels
            s = s + px.sum(1); ss = ss + px @ px.T; n += px.shape[1]
        mean = s / n
        return cls(mean, ss / n - np.outer(mean, mean))

    def for_matrix(self, M: np.ndarray):
        """Mean/std of the bands produced by response matrix M (B, C)."""
        mean = M @ self.mean
        std = np.sqrt(np.maximum(np.einsum("bc,cd,bd->b", M, self.cov, M), 1e-12))
        return mean, std

    def state(self):
        return {"mean": self.mean.tolist(), "cov": self.cov.tolist()}

    @classmethod
    def from_state(cls, d):
        return cls(np.array(d["mean"]), np.array(d["cov"]))


# ---------------------------------------------------------------- rendering on the GPU
def full_camera(wavelengths: np.ndarray) -> CameraProfile:
    """The hyperspectral sensor itself as a 'camera' (one band per source band)."""
    wl = np.asarray(wavelengths, float)
    step = float(np.median(np.diff(wl)))
    return CameraProfile.from_lists("hyperspectral_204", wl.tolist(), [step] * len(wl), type="hyperspectral")


class GPURenderer:
    def __init__(self, stats: SpectralStats, wavelengths: np.ndarray, device):
        self.stats, self.wl, self.dev = stats, np.asarray(wavelengths, float), device
        self._cache = {}

    def _prep(self, camera: CameraProfile, nominal: Optional[CameraProfile], to_full: bool):
        """Matrices for a camera. `nominal` = what the user believes the camera is (drift unknown)."""
        nominal = nominal or camera
        key = (tuple(camera.centers), tuple(camera.fwhms), tuple(nominal.centers), to_full)
        if key not in self._cache:
            n = len(self.wl)
            if camera.type == "hyperspectral" and len(camera) == n:
                M = np.eye(n)
            else:
                M = srf_matrix(self.wl, camera)
            if to_full:
                W = back_to_full_matrix(nominal, self.wl)          # (C, B)
                mean, std = self.stats.mean, np.sqrt(np.diag(self.stats.cov))
            else:
                W = None
                Mn = np.eye(n) if (nominal.type == "hyperspectral" and len(nominal) == n) else srf_matrix(self.wl, nominal)
                mean, std = self.stats.for_matrix(Mn)              # normalisation from the nominal datasheet
            t = lambda a: None if a is None else torch.as_tensor(a, dtype=torch.float32, device=self.dev)
            self._cache[key] = (t(M), t(W), t(mean)[:, None, None], t(std)[:, None, None])
        return self._cache[key]

    def render(self, x: torch.Tensor, camera: CameraProfile, nominal: Optional[CameraProfile] = None,
               to_full: bool = False, radiometric: float = 0.0, gen: Optional[torch.Generator] = None,
               gain=(0.8, 1.25), offset=0.02, snr_db=(20.0, 50.0)) -> torch.Tensor:
        """x: raw reflectance (N, C, H, W) on the device. radiometric = probability per sample."""
        M, W, mean, std = self._prep(camera, nominal, to_full)
        mask = (x.sum(1, keepdim=True) > 0).float()
        y = torch.einsum("bc,nchw->nbhw", M, x)
        if radiometric > 0:
            N, B = y.shape[:2]
            kw = dict(device=y.device, generator=gen)
            on = (torch.rand(N, 1, 1, 1, **kw) < radiometric).float()
            g = gain[0] + (gain[1] - gain[0]) * torch.rand(N, B, 1, 1, **kw)
            o = offset * (2 * torch.rand(N, B, 1, 1, **kw) - 1)
            snr = snr_db[0] + (snr_db[1] - snr_db[0]) * torch.rand(N, 1, 1, 1, **kw)
            yp = y * g + o
            rms = yp.pow(2).mean(dim=(1, 2, 3), keepdim=True).sqrt()
            yp = yp + torch.randn(yp.shape, **kw) * rms / 10 ** (snr / 20)
            y = on * yp + (1 - on) * y
        if W is not None:
            y = torch.einsum("cb,nbhw->nchw", W, y)
        y = y.clamp_min(1e-6)
        return ((y - mean) / std) * mask


# ---------------------------------------------------------------- model
def fourier_features(v: torch.Tensor, n_freq: int = 16) -> torch.Tensor:
    """v: (B,) values roughly in [0, 1] -> (B, 2*n_freq + 1)."""
    f = (2.0 ** torch.arange(n_freq, device=v.device, dtype=v.dtype)) * math.pi
    a = v[:, None] * f[None]
    return torch.cat([v[:, None], torch.sin(a), torch.cos(a)], 1)


class WavelengthStem(nn.Module):
    def __init__(self, rgb_weight: torch.Tensor, stride=2, padding=3, hidden=128, n_freq=16):
        super().__init__()
        out_ch, _, k, _ = rgb_weight.shape
        self.out_ch, self.k, self.stride, self.padding, self.n_freq = out_ch, k, stride, padding, n_freq
        self.register_buffer("rgb", rgb_weight.detach().clone())          # (out, 3, k, k)
        self.mlp = nn.Sequential(nn.Linear(2 * (2 * n_freq + 1), hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, out_ch * k * k))
        nn.init.zeros_(self.mlp[-1].weight); nn.init.zeros_(self.mlp[-1].bias)

    def base_filters(self, centers: torch.Tensor) -> torch.Tensor:
        """Pretrained RGB filters interpolated at each band centre (nearest outside 460-610 nm)."""
        c = torch.tensor(sorted(RGB_CENTRES), device=centers.device)          # 460, 540, 610 -> B, G, R
        filt = self.rgb[:, [2, 1, 0]]                                         # reorder to B, G, R
        lam = centers.clamp(c[0], c[-1])
        idx = torch.searchsorted(c, lam).clamp(1, len(c) - 1)
        lo, hi = idx - 1, idx
        t = ((lam - c[lo]) / (c[hi] - c[lo]))[None, :, None, None]
        return filt[:, lo] * (1 - t) + filt[:, hi] * t                      # (out, B, k, k)

    def forward(self, x, centers, fwhms):
        B = x.shape[1]
        feat = torch.cat([fourier_features((centers - 350.0) / 700.0, self.n_freq),
                          fourier_features(fwhms / 100.0, self.n_freq)], 1)
        delta = self.mlp(feat).view(B, self.out_ch, self.k, self.k).permute(1, 0, 2, 3)
        rms = self.rgb.pow(2).mean().sqrt()
        W = (self.base_filters(centers) + rms * delta) * (3.0 / B)   # delta = 0 at start
        return F.conv2d(x, W, stride=self.stride, padding=self.padding)


class SVCTNet(nn.Module):
    def __init__(self, n_classes=4, n_reg=4, backbone="resnet18", pretrained=True, dropout=0.2, upscale=1.0):
        super().__init__()
        import timm
        self.upscale = upscale
        self.backbone = timm.create_model(backbone, pretrained=pretrained, num_classes=0, global_pool="avg")
        conv = self.backbone.conv1
        if conv.in_channels != 3:
            raise ValueError("SVCTNet expects a ResNet-style backbone with an RGB conv1")
        self.stem = WavelengthStem(conv.weight.data, stride=conv.stride, padding=conv.padding)
        self.backbone.conv1 = nn.Identity()
        d = self.backbone.num_features
        self.drop = nn.Dropout(dropout)
        self.cls = nn.Linear(d, n_classes)
        self.reg = nn.Linear(d, n_reg)

    def forward(self, x, centers, fwhms):
        if self.upscale != 1.0:
            x = F.interpolate(x, scale_factor=self.upscale, mode="bilinear", align_corners=False)
        z = self.drop(self.backbone(self.stem(x, centers, fwhms)))
        return self.cls(z), self.reg(z)


def consistency_loss(out1, out2):
    """Symmetric KL between cultivar predictions + MSE between regression outputs of two views."""
    (l1, r1), (l2, r2) = out1, out2
    p1, p2 = F.log_softmax(l1.float(), 1), F.log_softmax(l2.float(), 1)
    kl = 0.5 * (F.kl_div(p1, p2, log_target=True, reduction="batchmean") +
                F.kl_div(p2, p1, log_target=True, reduction="batchmean"))
    return kl + F.mse_loss(r1.float(), r2.float())


def camera_tensors(camera: CameraProfile, device):
    return (torch.tensor(camera.centers, dtype=torch.float32, device=device),
            torch.tensor(camera.fwhms, dtype=torch.float32, device=device))


def geo_aug(cube: np.ndarray, rng: np.random.Generator, max_shift: int = 6) -> np.ndarray:
    """Flips + small shift of a RAW cube (spectral/radiometric changes happen on the GPU)."""
    if rng.random() < 0.5:
        cube = cube[:, :, ::-1]
    if rng.random() < 0.5:
        cube = cube[:, ::-1, :]
    dy, dx = rng.integers(-max_shift, max_shift + 1, 2)
    return np.ascontiguousarray(np.roll(cube, (int(dy), int(dx)), axis=(1, 2)))


# ---------------------------------------------------------------- round 2: structured virtual cameras
_MS_SLOTS = [(440.0, 500.0), (520.0, 580.0), (630.0, 690.0), (700.0, 750.0), (760.0, 900.0)]  # B, G, R, RE, NIR


def sample_structured_camera(rng: np.random.Generator, held_out: Sequence[CameraProfile] = (), tol_nm: float = 15.0,
                             wl_range=(400.0, 1000.0), max_tries: int = 200,
                             weights=(1 / 3, 1 / 3, 1 / 3)) -> CameraProfile:
    """Mixture of camera families, so that common real camera *types* are practised without copying any
    real camera (anything within tol_nm of a held-out camera is rejected):
      1/3 'rgb-like'  3-4 wide bands in the visible (optionally one NIR band)
      1/3 'multispectral'  4-6 bands from B/G/R/red-edge/NIR slots with jittered centres, 10-60 nm wide
      1/3 'random'    the round-1 sampler (3-12 bands anywhere, 10-100 nm)
    """
    from .simulate import _too_close, sample_virtual_camera
    lo, hi = wl_range
    for _ in range(max_tries):
        kind = rng.choice(3, p=np.asarray(weights, float) / np.sum(weights))  # 0 rgb-like, 1 multispectral, 2 random
        if kind == 2:
            return sample_virtual_camera(rng, held_out=held_out, fwhm_range=(10.0, 100.0), margin_scale=0.6)
        if kind == 0:
            c = np.sort(rng.uniform([430, 510, 580], [490, 570, 650]))
            w = rng.uniform(40.0, 100.0, 3)
            if rng.random() < 0.3:
                c = np.append(c, rng.uniform(780, 900)); w = np.append(w, rng.uniform(30, 100))
        else:
            k = int(rng.integers(4, 6))
            slots = np.sort(rng.choice(len(_MS_SLOTS), size=k, replace=False))
            c = np.array([rng.uniform(*_MS_SLOTS[s]) for s in slots])
            w = rng.uniform(10.0, 60.0, k)
        ok = (c - 0.6 * w > lo) & (c + 0.6 * w < hi)
        if not ok.all():
            continue
        if any(_too_close(c, h, tol_nm) for h in held_out if len(h) == len(c)):
            continue
        return CameraProfile.from_lists("virtual", c.tolist(), w.tolist(), type="virtual")
    raise RuntimeError("could not sample a structured camera")


def distill_loss(student, teacher, T: float = 2.0):
    """Student (camera view) learns from the detached full-band teacher view."""
    (ls, rs), (lt, rt) = student, teacher
    kl = F.kl_div(F.log_softmax(ls.float() / T, 1), F.log_softmax(lt.detach().float() / T, 1),
                  log_target=True, reduction="batchmean") * T * T
    return kl + F.mse_loss(rs.float(), rt.detach().float())
