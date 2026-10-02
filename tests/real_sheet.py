"""Render contact sheets of every animation for a real character image.
usage: python tests/real_sheet.py CHARACTER.png [BACKGROUND.png]"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animator.core.character import Action, Character  # noqa: E402
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from render_sheet import sheet  # noqa: E402

if __name__ == "__main__":
    cut = prepare_cutout(read_image(sys.argv[1]))
    j, _ = detect_joints(cut)
    scene = Scene()
    if len(sys.argv) > 2:
        scene.set_background(read_image(sys.argv[2]))
    ch = Character(Rig(cut, j), pos=(0.5, 0.95), height=0.8)
    scene.characters.append(ch)
    name = Path(sys.argv[1]).stem.split()[0]
    for clip in ["wave", "wave_other_arm", "walk", "jump"]:
        ch.actions = [Action(clip, target=(0.9, 0.95))] if clip == "walk" else [Action(clip)]
        d = scene.lib[clip].duration
        times = np.linspace(0, d, 8, endpoint=False) if clip != "walk" else np.linspace(0, 1, 8, endpoint=False)
        sheet(scene, times, f"test_output/{name}_{clip}.png", w=480, h=270, cols=4)
