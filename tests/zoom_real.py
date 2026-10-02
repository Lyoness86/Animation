"""Zoomed frames of one real character doing each animation.
usage: python tests/zoom_real.py CHARACTER.png [OUT.png]"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animator.core.character import Action, Character  # noqa: E402
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene, to_bgr_over  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402

SHOTS = [("wave_other_arm", 1.0), ("wave", 1.0), ("walk", 0.25), ("walk", 0.75), ("jump", 0.3), ("jump", 0.7)]

if __name__ == "__main__":
    cut = prepare_cutout(read_image(sys.argv[1]))
    j, _ = detect_joints(cut)
    s = Scene()
    ch = Character(Rig(cut, j), pos=(0.5, 0.98), height=0.95)
    s.characters.append(ch)
    tiles = []
    for clip, t in SHOTS:
        ch.actions = [Action(clip, target=(0.9, 0.98))] if clip == "walk" else [Action(clip)]
        f, boxes = s.render(t, 1920, 1080)
        img = to_bgr_over(f, (200, 200, 200))
        x0 = int(max(min(boxes[id(ch)][0], 1920 - 700), 0))
        tiles.append(img[:, x0:x0 + 700])
    out = sys.argv[2] if len(sys.argv) > 2 else "test_output/zoom_real.png"
    cv2.imwrite(out, cv2.resize(np.hstack(tiles), None, fx=0.6, fy=0.6, interpolation=cv2.INTER_AREA))
