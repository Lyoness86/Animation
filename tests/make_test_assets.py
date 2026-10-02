"""Creates placeholder assets: a glass prop. (python tests/make_test_assets.py)"""
from pathlib import Path

import cv2
import numpy as np


def make_glass(w=120, h=200):
    img = np.zeros((h, w, 4), np.uint8)
    body = np.array([[12, 8], [w - 12, 8], [w - 22, h - 10], [22, h - 10]], np.int32)
    drink = np.array([[17, 60], [w - 17, 60], [w - 22, h - 14], [22, h - 14]], np.int32)
    cv2.fillConvexPoly(img, body, (235, 225, 215, 90))           # glass, see-through
    cv2.fillConvexPoly(img, drink, (40, 140, 230, 235))            # orange drink
    cv2.line(img, (20, 65), (w - 20, 65), (200, 240, 255, 255), 3)  # drink surface
    cv2.polylines(img, [body], True, (250, 250, 250, 230), 4, cv2.LINE_AA)
    cv2.line(img, (30, 20), (36, h - 30), (255, 255, 255, 160), 6, cv2.LINE_AA)  # shine
    return img


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "assets" / "props" / "glass.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), make_glass())
    print("wrote", out)
