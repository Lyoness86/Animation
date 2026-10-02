"""Background removal: turns a character on a plain-colour background (blue,
green, yellow...) into a transparent (BGRA) cut-out, cropped to the character."""
import cv2
import numpy as np

WORK_HEIGHT = 900  # characters are processed at (at most) this height in pixels


def _border_pixels(img, width=6):
    return np.concatenate([
        img[:width].reshape(-1, 3), img[-width:].reshape(-1, 3),
        img[:, :width].reshape(-1, 3), img[:, -width:].reshape(-1, 3),
    ])


def remove_background(bgr):
    """Remove a plain single-colour background (blue, green, yellow...).
    The background colour is measured from the image border. Returns BGRA."""
    img = bgr.astype(np.float32) / 255.0
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    border = _border_pixels(lab)
    key_lab = np.median(border, axis=0)
    key_bgr = np.median(_border_pixels(img), axis=0)

    # Colour distance to the background (lightness counts less, so shadows
    # on the backdrop still count as background).
    w = np.array([0.5, 1.0, 1.0], np.float32)
    dist = np.sqrt((((lab - key_lab) * w) ** 2).sum(axis=2))
    noise = np.sqrt((((border - key_lab) * w) ** 2).sum(axis=1))
    lo = max(np.percentile(noise, 99) * 1.2, 6.0)
    hi = lo + 22.0
    alpha = np.clip((dist - lo) / (hi - lo), 0.0, 1.0)

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

    # Shrink the edge by ~1px to cut away the coloured fringe, then soften.
    a8 = (alpha * 255).astype(np.uint8)
    a8 = cv2.erode(a8, np.ones((3, 3), np.uint8))
    a8 = cv2.GaussianBlur(a8, (3, 3), 0)
    alpha = a8.astype(np.float32) / 255.0

    # Despill: in semi-transparent edge pixels (hair!), take the background
    # colour back out of the mix: fg = (pixel - (1-a)*bg) / a.
    edge = (alpha > 0.02) & (alpha < 0.98)
    a3 = np.maximum(alpha[..., None], 0.15)
    unmixed = np.clip((img - (1 - a3) * key_bgr) / a3, 0, 1)
    img = np.where(edge[..., None], unmixed, img)
    out = np.dstack([img, alpha])
    return np.clip(out * 255 + 0.5, 0, 255).astype(np.uint8)


remove_blue_screen = remove_background  # old name


def prepare_cutout(img):
    """Accept BGR (plain background) or BGRA (already transparent). Returns a BGRA
    cut-out cropped to the character with a small margin and scaled to at
    most WORK_HEIGHT tall."""
    if img.shape[2] == 4 and (img[..., 3] < 250).mean() > 0.01:
        rgba = img.copy()
    else:
        rgba = remove_background(img[..., :3])

    ys, xs = np.nonzero(rgba[..., 3] > 20)
    if len(xs) == 0:
        raise ValueError("No character found - is the background one plain colour?")
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
