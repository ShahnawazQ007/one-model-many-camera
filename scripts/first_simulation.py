"""Show the camera simulator on a REAL hyperspectral image (first look, Step 1 on real data).

Usage:
    python scripts/first_simulation.py <data folder or cube file> [out.png]

If the file has no wavelength list, Specim IQ sampling is assumed:
204 bands, 397-1004 nm (approximate; replace with the dataset's own list when found).
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam import load_camera, simulate, srf_matrix  # noqa: E402
from anycam.io import find_cubes, load_cube  # noqa: E402

SPECIM_IQ_WL = np.linspace(397.0, 1004.0, 204)


def stretch(img):
    lo, hi = np.percentile(img, [2, 98])
    return np.clip((img - lo) / (hi - lo + 1e-9), 0, 1)


def rgb_from(cube, wl, targets=(640, 550, 460)):
    idx = [int(np.argmin(np.abs(wl - t))) for t in targets]
    return np.dstack([stretch(cube[i]) for i in idx])


def main(src, out="first_simulation.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    src = pathlib.Path(src)
    path = src if src.is_file() else (find_cubes(src, 1) or [None])[0]
    if path is None:
        sys.exit(f"No cube files (.hdr/.npy/.npz/.tif) found under {src}")
    cube, wl = load_cube(path)
    assumed = wl is None
    if assumed:
        if cube.shape[0] != 204:
            sys.exit(f"{path.name}: {cube.shape[0]} bands and no wavelength list; pass the dataset's wavelengths")
        wl = SPECIM_IQ_WL
    print(f"{path.name}: {cube.shape[0]} bands x {cube.shape[1]} x {cube.shape[2]} px, "
          f"{wl.min():.0f}-{wl.max():.0f} nm{' (ASSUMED wavelengths)' if assumed else ''}, "
          f"values {cube.min():.3g}..{cube.max():.3g}")

    mica = load_camera("micasense_rededge_mx")
    phone = load_camera("generic_rgb")
    sim_m = simulate(cube, wl, mica, axis=0)
    sim_p = simulate(cube, wl, phone, axis=0)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(2, 4, 1); ax.imshow(rgb_from(cube, wl)); ax.set_title(f"Lab camera ({cube.shape[0]} bands)\nshown as colour"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 2); ax.imshow(np.dstack([stretch(sim_p[i]) for i in (2, 1, 0)])); ax.set_title("Simulated phone (3 bands)"); ax.axis("off")
    names = [b.name for b in mica.bands]
    ax = fig.add_subplot(2, 4, 3); ax.imshow(stretch(sim_m[names.index("red_edge")]), cmap="gray"); ax.set_title("Simulated drone: red-edge band"); ax.axis("off")
    ax = fig.add_subplot(2, 4, 4); ax.imshow(stretch(sim_m[names.index("nir")]), cmap="gray"); ax.set_title("Simulated drone: NIR band"); ax.axis("off")

    ax = fig.add_subplot(2, 1, 2)
    mean_spec = cube.reshape(cube.shape[0], -1).mean(1)
    ax.plot(wl, mean_spec, color="0.2", lw=2, label=f"Lab camera: {cube.shape[0]} numbers per pixel")
    M = srf_matrix(wl, mica)
    for b, row, val in zip(mica.bands, M, sim_m.reshape(len(mica), -1).mean(1)):
        ax.fill_between(wl, 0, row / row.max() * mean_spec.max(), alpha=0.15, color="tab:blue")
        ax.plot(b.center, val, "o", color="tab:orange", ms=10)
        ax.annotate(b.name, (b.center, val), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.plot([], [], "o", color="tab:orange", label="Simulated drone camera: 5 numbers per pixel")
    ax.fill_between([], [], color="tab:blue", alpha=0.15, label="Drone camera band windows")
    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Mean value over the leaf"); ax.legend(loc="upper left")
    fig.suptitle(f"Camera simulation on a real image: {path.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    print("saved", out)


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    main(*sys.argv[1:])
