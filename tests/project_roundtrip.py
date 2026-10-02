"""Build a full project (video background, real characters, views, prop,
actions), save it, reopen it, and check the frames are identical."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animator.core.character import Action, Character, Prop  # noqa: E402
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.project import load_project, save_project  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from view_sheet import build_view_character  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def build():
    s = Scene()
    s.set_background_video(ROOT / "test_output/bg_test.mp4")
    for f, pos in [("karen no drink.png", (0.75, 0.95)), ("Leah no drink.png", (0.15, 0.95))]:
        cut = prepare_cutout(read_image(ROOT / f))
        j, _ = detect_joints(cut)
        s.characters.append(Character(Rig(cut, j, f.split()[0]), pos=pos, height=0.5))
    karen, leah = s.characters
    karen.props.append(Prop(read_image(ROOT / "assets/props/glass.png"), "Glass", hand="l", size=0.13, layer=1))
    karen.actions = [Action("idle", 0.5), Action("drink_other_hand"), Action("nod")]
    leah.mirrored = True
    leah.actions = [Action("walk", target=(0.45, 0.95)), Action("stop"), Action("dance")]
    vc = build_view_character()
    vc.height = 0.45
    s.characters.insert(0, vc)  # back-most layer
    return s


if __name__ == "__main__":
    s = build()
    out = ROOT / "test_output/test.puppet"
    t0 = time.time()
    save_project(s, out)
    print("saved", out.stat().st_size // 1024, "KB in", round(time.time() - t0, 2), "s")
    t0 = time.time()
    s2 = load_project(out)
    print("loaded in", round(time.time() - t0, 2), "s")
    assert [c.name for c in s2.characters] == [c.name for c in s.characters]
    worst = 0.0
    for t in [0.0, 1.0, 2.2, 3.5, 5.0]:
        a, _ = s.render(t, 640, 360)
        b, _ = s2.render(t, 640, 360)
        worst = max(worst, float(np.abs(a - b).max()))
    print("max pixel difference after reopen:", worst)
    assert worst < 1e-4, worst
    print("ROUND TRIP OK")
