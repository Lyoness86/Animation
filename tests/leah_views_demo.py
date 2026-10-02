"""Builds Leah with front/side/back views on the TDP background, saves
test_output/leah_views.puppet and renders a side-walk strip + part overlay."""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]
from animator.core.character import Action, Character  # noqa: E402
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.project import save_project  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene, to_bgr_over  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from debug_rig import overlay  # noqa: E402

s = Scene()
s.set_background(read_image(ROOT / "tdp 3d.png"))
front = prepare_cutout(read_image(ROOT / "Leah front.jpg"))
jf, _ = detect_joints(front)
rigs = {"front": Rig(front, jf, "Leah")}
for f, v in [("Leah side.jpg", "side"), ("Leah back.jpg", "back")]:
    cut = prepare_cutout(read_image(ROOT / f))
    j, _ = detect_joints(cut, reference=(jf, front), view=v)
    rigs[v] = Rig(cut, j, "Leah", faces_left=(v == "side"), profile=(v == "side"))
r = rigs["side"]
cv2.imwrite("test_output/side_arm.png", overlay(r, r.joints))
ch = Character(rigs["front"], pos=(0.2, 0.9), height=0.6, views=rigs)
s.characters.append(ch)
ch.actions = [Action("walk", target=(0.55, 0.9)), Action("stop"), Action("wave"), Action("point"),
              Action("dance"), Action("walk", target=(0.25, 0.9)), Action("stop")]
save_project(s, "test_output/leah_views.puppet")
tiles = []
for t in [0.9, 1.15, 1.4, 1.65, 1.9]:
    f, b = s.render(t, 1920, 1080)
    img = to_bgr_over(f)
    bx = b[id(ch)]
    cx = (bx[0] + bx[2]) // 2
    x0, y0 = int(max(cx - 200, 0)), int(max(bx[1] - 20, 0))
    tiles.append(cv2.resize(img[y0:y0 + 680, x0:x0 + 400], (400, 680)))
cv2.imwrite("test_output/sidewalk.png", cv2.resize(np.hstack(tiles), None, fx=0.8, fy=0.8))
