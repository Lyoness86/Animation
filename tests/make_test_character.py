"""Draws a simple shaded cartoon character on a blue screen for testing."""
import cv2
import numpy as np


def _shaded_ellipse(img, center, axes, colour, angle=0):
    mask = np.zeros(img.shape[:2], np.uint8)
    cv2.ellipse(mask, center, axes, angle, 0, 360, 255, -1)
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    # light from top-left
    d = ((xx - (center[0] - axes[0] * 0.4)) ** 2 + (yy - (center[1] - axes[1] * 0.4)) ** 2) ** 0.5
    shade = np.clip(1.15 - d / (max(axes) * 2.2), 0.55, 1.15)[..., None]
    col = np.clip(np.array(colour, np.float32) * shade, 0, 255).astype(np.uint8)
    img[mask > 0] = col[mask > 0]


def _limb(img, p0, p1, width, colour):
    p0, p1 = np.array(p0, float), np.array(p1, float)
    c = ((p0 + p1) / 2).astype(int)
    L = np.linalg.norm(p1 - p0)
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    _shaded_ellipse(img, tuple(c), (int(L / 2 + width * 0.4), int(width / 2)), colour, ang)


def make_character(h=1000, w=700, shirt=(40, 60, 200), seed=0, bg=(230, 60, 20)):
    img = np.zeros((h, w, 3), np.uint8)
    img[:] = bg  # BGR; default blue screen (slightly uneven)
    rng = np.random.default_rng(seed)
    img = np.clip(img.astype(int) + rng.integers(-8, 8, img.shape), 0, 255).astype(np.uint8)
    cx = w // 2
    skin, pants, shoe = (150, 185, 235), (90, 60, 40), (40, 40, 40)
    # legs (gap between them)
    for s in (-1, 1):
        _limb(img, (cx + s * 45, 600), (cx + s * 50, 760), 70, pants)
        _limb(img, (cx + s * 50, 760), (cx + s * 52, 920), 58, pants)
        _shaded_ellipse(img, (cx + s * 60, 945), (48, 24), shoe)
    # torso
    _shaded_ellipse(img, (cx, 470), (115, 170), shirt)
    # arms hanging at the sides with a gap
    for s in (-1, 1):
        _limb(img, (cx + s * 120, 345), (cx + s * 150, 500), 52, shirt)
        _limb(img, (cx + s * 150, 500), (cx + s * 158, 620), 44, skin)
        _shaded_ellipse(img, (cx + s * 160, 645), (28, 30), skin)
    # head
    _shaded_ellipse(img, (cx, 200), (125, 135), skin)
    _shaded_ellipse(img, (cx, 95), (115, 60), (30, 50, 90))  # hair
    for s in (-1, 1):
        cv2.circle(img, (cx + s * 45, 205), 16, (255, 255, 255), -1)
        cv2.circle(img, (cx + s * 45, 208), 8, (20, 20, 20), -1)
    cv2.ellipse(img, (cx, 265), (40, 18), 0, 0, 180, (60, 60, 160), 4)
    return img


if __name__ == "__main__":
    import sys
    cv2.imwrite(sys.argv[1] if len(sys.argv) > 1 else "test_character.png", make_character())
