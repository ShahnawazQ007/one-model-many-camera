"""Print the camera registry."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from anycam import list_cameras, load_camera

for name in list_cameras():
    cam = load_camera(name)
    bands = ", ".join(f"{b.name} {b.center:g}/{b.fwhm:g}" for b in cam.bands)
    print(f"{name:26s} {len(cam):2d} bands  [{bands}]")
