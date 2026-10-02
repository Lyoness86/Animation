"""One frame from the middle of every animation for a character.
usage: python tests/library_sheet.py CHARACTER.png [OUT.png] [clip,clip,...]"""
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

# clip, time to show (seconds into the clip), optional action before it
SHOTS = [("nod", 0.25), ("shake_head", 0.45), ("look_left", 0.8), ("look_right", 0.8),
         ("point", 1.0), ("clap", 0.7), ("clap", 0.9), ("dance", 0.5), ("dance", 1.0),
         ("sit", 1.2), ("sit_idle", 1.0), ("stand", 0.6), ("celebrate", 1.0), ("laugh", 1.0),
         ("shrug", 0.8)]

if __name__ == "__main__":
    cut = prepare_cutout(read_image(sys.argv[1]))
    j, _ = detect_joints(cut)
    s = Scene()
    ch = Character(Rig(cut, j), pos=(0.5, 0.98), height=0.9)
    s.characters.append(ch)
    tiles = []
    only = set(sys.argv[3].split(",")) if len(sys.argv) > 3 else None
    for clip, t in SHOTS:
        if only and clip not in only:
            continue
        if clip == "sit_idle":
            ch.actions, t = [Action("sit"), Action("idle", 2.0)], 1.2 + t
        elif clip == "stand":
            ch.actions, t = [Action("sit"), Action("stand")], 1.2 + t
        else:
            ch.actions = [Action(clip)]
        f, boxes = s.render(t, 1920, 1080)
        img = to_bgr_over(f, (200, 200, 200))
        cx = (boxes[id(ch)][0] + boxes[id(ch)][2]) // 2
        x0 = int(min(max(cx - 300, 0), 1920 - 600))
        tile = img[:, x0:x0 + 600].copy()
        cv2.putText(tile, f"{clip} {t:.1f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
        tiles.append(tile)
    while len(tiles) % 5:
        tiles.append(np.full_like(tiles[0], 200))
    rows = [np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)]
    out = sys.argv[2] if len(sys.argv) > 2 else "test_output/library.png"
    cv2.imwrite(out, cv2.resize(np.vstack(rows), None, fx=0.4, fy=0.4, interpolation=cv2.INTER_AREA))
