"""Render a character holding the placeholder glass through the drink
animation. usage: python tests/prop_sheet.py CHARACTER.png [hand l|r] [OUT]"""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animator.core.character import Action, Character, Prop  # noqa: E402
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene, to_bgr_over  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    hand = sys.argv[2] if len(sys.argv) > 2 else "l"
    cut = prepare_cutout(read_image(sys.argv[1]))
    j, _ = detect_joints(cut)
    s = Scene()
    ch = Character(Rig(cut, j), pos=(0.5, 0.98), height=0.95)
    ch.props.append(Prop(read_image(ROOT / "assets/props/glass.png"), "Glass", hand=hand))
    s.characters.append(ch)
    ch.actions = [Action("drink" if hand == "r" else "drink_other_hand"), Action("wave" if hand == "r" else "wave_other_arm")]
    tiles = []
    for t in [0.0, 0.5, 1.0, 1.5, 2.3, 3.6]:
        f, boxes = s.render(t, 1920, 1080)
        img = to_bgr_over(f, (200, 200, 200))
        x0 = int(max(min(boxes[id(ch)][0] - 40, 1920 - 600), 0))
        tiles.append(img[:, x0:x0 + 600])
    out = sys.argv[3] if len(sys.argv) > 3 else "test_output/prop_sheet.png"
    cv2.imwrite(out, cv2.resize(np.hstack(tiles), None, fx=0.55, fy=0.55, interpolation=cv2.INTER_AREA))
