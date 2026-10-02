"""Render a 2-character test scene and save contact sheets of frames."""
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animator.core.character import Action, Character  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene, to_bgr_over  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from make_test_character import make_character  # noqa: E402


def build_scene():
    scene = Scene()
    for i, shirt in enumerate([(40, 60, 200), (60, 160, 40)]):
        cut = prepare_cutout(make_character(shirt=shirt, seed=i))
        j, _ = detect_joints(cut)
        scene.characters.append(Character(Rig(cut, j, f"Char{i + 1}"), pos=(0.2 + 0.5 * i, 0.92), height=0.6))
    a, b = scene.characters
    a.actions = [Action("walk", target=(0.45, 0.92)), Action("stop"), Action("wave")]
    b.actions = [Action("idle", 1.0), Action("jump"), Action("walk", target=(0.85, 0.92)), Action("stop")]
    return scene


def sheet(scene, times, path, w=480, h=270, cols=4):
    tiles = []
    for t in times:
        f, _ = scene.render(t, w, h)
        img = to_bgr_over(f)
        cv2.putText(img, f"t={t:.2f}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
    cv2.imwrite(path, np.vstack(rows))


if __name__ == "__main__":
    scene = build_scene()
    Path("test_output").mkdir(exist_ok=True)
    print("duration", scene.duration())
    t0 = time.time()
    for t in np.arange(0, 3, 0.1):
        scene.render(t, 1920, 1080)
    print("1080p ms/frame", (time.time() - t0) / 30 * 1000)
    t0 = time.time()
    for t in np.arange(0, 3, 0.1):
        scene.render(t, 960, 540)
    print("540p ms/frame", (time.time() - t0) / 30 * 1000)
    sheet(scene, np.arange(0, 2.4, 0.15), "test_output/sheet_walk.png")
    sheet(scene, np.arange(1.0, 2.4, 0.1), "test_output/sheet_jump.png")
    sheet(scene, np.arange(4.4, 7.6, 0.2), "test_output/sheet_wave.png")
