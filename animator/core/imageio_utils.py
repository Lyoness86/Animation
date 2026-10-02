"""Image loading/saving that works with non-ASCII Windows paths."""
import cv2
import numpy as np


def read_image(path):
    """Read an image as BGR or BGRA uint8 (keeps alpha if present)."""
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read image: {path}")
    if img.dtype != np.uint8:
        img = (img / 256).astype(np.uint8)
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def write_png(path, img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    buf.tofile(str(path))
