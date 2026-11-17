"""Camera simulator and virtual-camera sampler (SVCT).

Band value = sum_l R(l) S_b(l) / sum_l S_b(l), with S_b a Gaussian of the band's
centre and FWHM (or a measured response curve, later).

Array convention: the spectral axis is given explicitly (`axis`), default -1.
  spectra:  (N, C)                 axis=-1
  images:   (C, H, W) or (B,C,H,W)  axis=0 / axis=1
"""
from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np

from .cameras import Band, CameraProfile

FWHM_TO_SIGMA = 1.0 / (2.0 * math.sqrt(2.0 * math.log(2.0)))  # ~0.4247
RED_EDGE = (700.0, 740.0)


# ---------------------------------------------------------------- response functions
def gaussian_srf(wavelengths: np.ndarray, center: float, fwhm: float) -> np.ndarray:
    """Gaussian spectral response sampled at the source wavelengths (not normalised)."""
    sigma = max(fwhm * FWHM_TO_SIGMA, 1e-6)
    return np.exp(-0.5 * ((np.asarray(wavelengths, float) - center) / sigma) ** 2)


def band_coverage(wavelengths: np.ndarray, center: float, fwhm: float) -> float:
    """Fraction of the band's Gaussian mass that lies inside the source range."""
    wl = np.sort(np.asarray(wavelengths, float))
    half_step = 0.5 * float(np.median(np.diff(wl))) if wl.size > 1 else 0.0
    lo, hi = wl[0] - half_step, wl[-1] + half_step  # each source band covers +/- half a step
    sigma = max(fwhm * FWHM_TO_SIGMA, 1e-6)
    cdf = lambda x: 0.5 * (1.0 + math.erf((x - center) / (sigma * math.sqrt(2.0))))
    return cdf(hi) - cdf(lo)


def srf_matrix(wavelengths: np.ndarray, camera: CameraProfile,
               min_coverage: float = 0.9) -> np.ndarray:
    """(n_bands, C) matrix whose rows are normalised responses (each row sums to 1).

    For a band narrower than the source sampling, the nearest source band is used.
    """
    wl = np.asarray(wavelengths, float)
    if wl.ndim != 1 or wl.size < 2:
        raise ValueError("wavelengths must be a 1-D array with >= 2 entries")
    rows = []
    for b in camera.bands:
        cov = band_coverage(wl, b.center, b.fwhm)
        if cov < min_coverage:
            raise ValueError(
                f"Band {b.name} ({b.center} nm, FWHM {b.fwhm}) is only {cov:.0%} inside the "
                f"source range {wl.min():.0f}-{wl.max():.0f} nm")
        w = gaussian_srf(wl, b.center, b.fwhm)
        if w.sum() < 1e-8:  # band much narrower than sampling: take nearest source band
            w = np.zeros_like(wl)
            w[np.argmin(np.abs(wl - b.center))] = 1.0
        rows.append(w / w.sum())
    return np.stack(rows)


def simulate(cube: np.ndarray, wavelengths: np.ndarray, camera: CameraProfile,
             axis: int = -1, min_coverage: float = 0.9) -> np.ndarray:
    """Turn hyperspectral reflectance into what `camera` would record (noise-free)."""
    M = srf_matrix(wavelengths, camera, min_coverage)          # (B, C)
    x = np.moveaxis(np.asarray(cube, float), axis, -1)          # (..., C)
    if x.shape[-1] != M.shape[1]:
        raise ValueError(f"cube has {x.shape[-1]} bands but {M.shape[1]} wavelengths given")
    y = x @ M.T                                                 # (..., B)
    return np.moveaxis(y, -1, axis)


# ---------------------------------------------------------------- calibration errors
def perturb_camera(camera: CameraProfile, rng: np.random.Generator,
                   max_shift_nm: float = 10.0, fwhm_scale: float = 0.25) -> CameraProfile:
    """Same camera with each band's centre shifted and FWHM rescaled (manufacturing drift)."""
    bands = []
    for b in camera.bands:
        shift = rng.uniform(-max_shift_nm, max_shift_nm)
        scale = 1.0 + rng.uniform(-fwhm_scale, fwhm_scale)
        bands.append(Band(b.name, b.center + shift, max(b.fwhm * scale, 1.0)))
    return CameraProfile(name=f"{camera.name}+drift", bands=bands, type=camera.type,
                         source=camera.source, display_name=camera.display_name)


def shift_camera(camera: CameraProfile, shift_nm: float) -> CameraProfile:
    """Deterministic shift of all band centres (for the E2 sweeps: +/-2, 5, 10 nm)."""
    bands = [Band(b.name, b.center + shift_nm, b.fwhm) for b in camera.bands]
    return CameraProfile(name=f"{camera.name}{shift_nm:+g}nm", bands=bands, type=camera.type)


def apply_radiometric(x: np.ndarray, rng: np.random.Generator, axis: int = -1,
                      gain_range: Tuple[float, float] = (0.9, 1.1),
                      offset_range: Tuple[float, float] = (-0.02, 0.02),
                      snr_db: Optional[Tuple[float, float]] = (20.0, 50.0)) -> np.ndarray:
    """Per-band gain/offset error plus Gaussian sensor noise at a random SNR."""
    x = np.moveaxis(np.asarray(x, float), axis, -1)
    nb = x.shape[-1]
    gain = rng.uniform(*gain_range, size=nb)
    offset = rng.uniform(*offset_range, size=nb)
    y = x * gain + offset
    if snr_db is not None:
        snr = rng.uniform(*snr_db)
        signal_rms = np.sqrt(np.mean(y ** 2)) + 1e-12
        noise_std = signal_rms / (10 ** (snr / 20.0))
        y = y + rng.normal(0.0, noise_std, size=y.shape)
    return np.moveaxis(y, -1, axis)


# ---------------------------------------------------------------- virtual cameras (SVCT)
def _too_close(centers: Sequence[float], held_out: CameraProfile, tol_nm: float) -> bool:
    """True if every sampled band lies within tol_nm of some band of a held-out camera."""
    ref = np.asarray(held_out.centers)
    return all(np.min(np.abs(ref - c)) <= tol_nm for c in centers)


def sample_virtual_camera(rng: np.random.Generator,
                          wl_range: Tuple[float, float] = (400.0, 1000.0),
                          n_bands: Tuple[int, int] = (3, 12),
                          fwhm_range: Tuple[float, float] = (10.0, 60.0),
                          p_red_edge: float = 0.5,
                          held_out: Iterable[CameraProfile] = (),
                          tol_nm: float = 15.0,
                          max_tries: int = 100) -> CameraProfile:
    """Draw a random camera for one mini-batch (leave-camera-out aware)."""
    lo, hi = wl_range
    held_out = list(held_out)
    for _ in range(max_tries):
        n = int(rng.integers(n_bands[0], n_bands[1] + 1))
        fwhm = rng.uniform(*fwhm_range, size=n)
        margin = fwhm  # keep each band's mass inside the source range
        centers = rng.uniform(lo + margin, hi - margin)
        if rng.random() < p_red_edge:
            centers[0] = rng.uniform(*RED_EDGE)
            fwhm[0] = min(fwhm[0], 30.0)
        order = np.argsort(centers)
        centers, fwhm = centers[order], fwhm[order]
        if any(_too_close(centers, h, tol_nm) for h in held_out if len(h) == n):
            continue
        return CameraProfile.from_lists("virtual", centers.tolist(), fwhm.tolist(),
                                        type="virtual")
    raise RuntimeError("Could not sample a camera far enough from the held-out cameras")
