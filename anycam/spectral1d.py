"""SVCT for point spectra (one spectrum per sample instead of an image).

Used by the real-camera experiments (Step 8): DeepHS fruit (FX10 -> real Corning camera) and Estrada
leaves (ASD spectrometer -> real MicaSense). The ideas are identical to the image version:
random virtual cameras + calibration errors during training; at test time the camera's bands are
either interpolated back to the training grid ('interp', plain MLP) or fed as tokens with their
wavelengths ('tok', wavelength-aware set model).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .camera_tf import back_to_full_matrix
from .cameras import CameraProfile
from .simulate import srf_matrix
from .svct import SpectralStats, fourier_features, sample_structured_camera


def grid_camera(wavelengths, name="full", fwhm: Optional[float] = None, type="hyperspectral") -> CameraProfile:
    wl = np.asarray(wavelengths, float)
    step = fwhm or float(np.median(np.diff(wl)))
    return CameraProfile.from_lists(name, wl.tolist(), [step] * len(wl), type=type)


class Spectral1D:
    """Rendering and normalisation for one training grid `wl` (the training sensor's wavelengths)."""

    def __init__(self, wl, train_spectra: np.ndarray):
        self.wl = np.asarray(wl, float)
        self.full = grid_camera(self.wl)
        x = np.asarray(train_spectra, np.float64)
        self.stats = SpectralStats(x.mean(0), np.cov(x, rowvar=False, bias=True))
        self._cache: Dict = {}

    def _M(self, cam: CameraProfile, src_wl=None):
        src = self.wl if src_wl is None else np.asarray(src_wl, float)
        key = ("M", tuple(cam.centers), tuple(cam.fwhms), len(src), float(src[0]))
        if key not in self._cache:
            if cam is self.full and src_wl is None:
                M = np.eye(len(self.wl))
            else:
                M = srf_matrix(src, cam)
            self._cache[key] = torch.tensor(M, dtype=torch.float32)
        return self._cache[key]

    def bands(self, x: torch.Tensor, cam: CameraProfile, src_wl=None, radio: float = 0.0,
              gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """Spectra (N, C_src) -> camera band values (N, B), optionally with calibration error."""
        y = x @ self._M(cam, src_wl).T
        if radio > 0:
            N, B = y.shape
            on = (torch.rand(N, 1, generator=gen) < radio).float()
            g = 0.8 + 0.45 * torch.rand(N, B, generator=gen)
            o = 0.02 * (2 * torch.rand(N, B, generator=gen) - 1)
            snr = 20 + 30 * torch.rand(N, 1, generator=gen)
            yp = y * g + o
            yp = yp + torch.randn(yp.shape, generator=gen) * yp.pow(2).mean(1, keepdim=True).sqrt() / 10 ** (snr / 20)
            y = on * yp + (1 - on) * y
        return y

    def model_input(self, y: torch.Tensor, nominal: CameraProfile, mode: str) -> torch.Tensor:
        """Camera band values (N, B) -> model input. interp: back to the training grid; tok: native bands."""
        if mode == "interp":
            if nominal is not self.full:
                key = ("W", tuple(nominal.centers), tuple(nominal.fwhms))
                if key not in self._cache:
                    self._cache[key] = torch.tensor(back_to_full_matrix(nominal, self.wl), dtype=torch.float32)
                y = y @ self._cache[key].T
            mean, std = self.stats.mean, np.sqrt(np.diag(self.stats.cov))
        else:
            Mn = np.eye(len(self.wl)) if nominal is self.full else srf_matrix(self.wl, nominal, min_coverage=0.0)
            mean, std = self.stats.for_matrix(Mn)
        return (y - torch.tensor(mean, dtype=torch.float32)) / torch.tensor(std, dtype=torch.float32)


class MLP(nn.Module):
    def __init__(self, d_in, heads: Sequence[int], hidden=256, drop=0.1):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Dropout(drop),
                                  nn.Linear(hidden, hidden), nn.GELU(), nn.Dropout(drop))
        self.heads = nn.ModuleList(nn.Linear(hidden, k) for k in heads)

    def forward(self, x, *_):
        h = self.body(x)
        return [hd(h) for hd in self.heads]


class TokenNet(nn.Module):
    """One token per band: (value, Fourier(centre), Fourier(width)) -> shared MLP -> mean + max pooling."""

    def __init__(self, heads: Sequence[int], hidden=128, n_freq=8, drop=0.1):
        super().__init__()
        self.n_freq = n_freq
        d_tok = 1 + 2 * (2 * n_freq + 1)
        self.tok = nn.Sequential(nn.Linear(d_tok, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.body = nn.Sequential(nn.Linear(2 * hidden, 256), nn.GELU(), nn.Dropout(drop))
        self.heads = nn.ModuleList(nn.Linear(256, k) for k in heads)

    def forward(self, x, centers, fwhms):
        N, B = x.shape
        f = torch.cat([fourier_features((centers - 350.0) / 700.0, self.n_freq),
                       fourier_features(fwhms / 100.0, self.n_freq)], 1)
        h = self.tok(torch.cat([x[..., None], f[None].expand(N, B, -1)], 2))
        h = self.body(torch.cat([h.mean(1), h.max(1).values], 1))
        return [hd(h) for hd in self.heads]


def cam_tensors(cam: CameraProfile):
    return (torch.tensor(cam.centers, dtype=torch.float32), torch.tensor(cam.fwhms, dtype=torch.float32))


def multitask_loss(outs, y_cls: Sequence[torch.Tensor], y_reg: torch.Tensor, n_cls_heads: int):
    loss = sum(F.cross_entropy(outs[k], y_cls[k]) for k in range(n_cls_heads))
    r = outs[n_cls_heads]
    m = torch.isfinite(y_reg)
    return loss + (F.mse_loss(r[m], y_reg[m]) if m.any() else r.sum() * 0)


def fit(model, make_batch, n, epochs, seed, n_cls_heads, batch=64, lr=1e-3, wd=1e-3):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            if len(b) < 4:
                continue
            x, extra, yc, yr = make_batch(b)
            loss = multitask_loss(model(x, *extra), yc, yr, n_cls_heads)
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    return model.eval()


def evaluate(outs, y_cls: Sequence[np.ndarray], y_reg_raw: np.ndarray, rmean, rstd, cls_names, reg_names):
    res = {}
    for k, name in enumerate(cls_names):
        res[f"{name}_acc"] = float((outs[k].argmax(1).numpy() == y_cls[k]).mean())
    r = outs[len(cls_names)].detach().numpy() * rstd + rmean
    for j, name in enumerate(reg_names):
        m = np.isfinite(y_reg_raw[:, j])
        t, pr = y_reg_raw[m, j], r[m, j]
        res[f"{name}_R2"] = float(1 - ((t - pr) ** 2).sum() / ((t - t.mean()) ** 2).sum())
    return res


def svct_camera(rng, full, held_out, p_full=0.3, weights=(0.45, 0.35, 0.2), wl_range=(400.0, 1000.0)):
    if rng.random() < p_full:
        return full
    return sample_structured_camera(rng, held_out=held_out, weights=weights, wl_range=wl_range)
