"""Blue-screen removal: turns a character on a blue background into a
transparent (BGRA) cut-out, cropped to the character."""
import cv2
import numpy as np

WORK_HEIGHT = 900  # characters are processed at (at most) this height in pixels


def _border_pixels(img, width=6):
    return np.concatenate([
        img[:width].reshape(-1, 3), img[-width:].reshape(-1, 3),
        img[:, :width].reshape(-1, 3), img[:, -width:].reshape(-1, 3),
    ])


def remove_blue_screen(bgr):
    """Return BGRA uint8. The key colour is estimated from the image border."""
    img = bgr.astype(np.float32) / 255.0
    b, g, r = img[..., 0], img[..., 1], img[..., 2]
    kb, kg, kr = np.median(_border_pixels(img), axis=0)

    # "Blue dominance": how much bluer than red/green a pixel is.
    dom = b - np.maximum(r, g)
    key_dom = max(kb - max(kr, kg), 0.15)
    lo, hi = key_dom * 0.25, key_dom * 0.65
    alpha = 1.0 - np.clip((dom - lo) / (hi - lo), 0.0, 1.0)

    # Remove specks: keep the large connected regions only.
    solid = (alpha > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(solid, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        keep = np.zeros(n, bool)
        keep[1:] = areas >= max(areas.max() * 0.02, 50)
        region = keep[labels].astype(np.uint8)
        region = cv2.dilate(region, np.ones((5, 5), np.uint8))
        alpha *= region

    # Shrink the edge by ~1px to cut away the blue fringe, then soften.
    a8 = (alpha * 255).astype(np.uint8)
    a8 = cv2.erode(a8, np.ones((3, 3), np.uint8))
    a8 = cv2.GaussianBlur(a8, (3, 3), 0)
    alpha = a8.astype(np.float32) / 255.0

    # Despill: near the edges, stop blue from exceeding red/green.
    edge = cv2.dilate((alpha < 0.98).astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    limit = np.maximum(r, g)
    b2 = np.where(edge & (b > limit), limit, b)
    out = np.dstack([b2, g, r, alpha])
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


def prepare_cutout(img):
    """Accept BGR (blue screen) or BGRA (already transparent). Returns a BGRA
    cut-out cropped to the character with a small margin and scaled to at
    most WORK_HEIGHT tall."""
    if img.shape[2] == 4 and (img[..., 3] < 250).mean() > 0.01:
        rgba = img.copy()
    else:
        rgba = remove_blue_screen(img[..., :3])

    ys, xs = np.nonzero(rgba[..., 3] > 20)
    if len(xs) == 0:
        raise ValueError("No character found - is the background blue?")
    h = ys.max() - ys.min() + 1
    m = int(h * 0.03) + 2
    y0, y1 = max(ys.min() - m, 0), min(ys.max() + m + 1, rgba.shape[0])
    x0, x1 = max(xs.min() - m, 0), min(xs.max() + m + 1, rgba.shape[1])
    rgba = rgba[y0:y1, x0:x1]
    if rgba.shape[0] > WORK_HEIGHT:
        s = WORK_HEIGHT / rgba.shape[0]
        rgba = cv2.resize(rgba, (max(1, round(rgba.shape[1] * s)), WORK_HEIGHT),
                          interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgba)
