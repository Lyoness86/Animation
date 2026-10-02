"""Test character with all five views: turns through them, walks in side
view and turns back. Saves a contact sheet."""
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animator.core.character import VIEW_NAMES, Action, Character  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.scene import Scene  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from make_test_character import make_view  # noqa: E402
from render_sheet import sheet  # noqa: E402


def build_view_character():
    rigs = {}
    for v in VIEW_NAMES:
        cut = prepare_cutout(make_view(v))
        j, _ = detect_joints(cut)
        rigs[v] = Rig(cut, j, "Viewtest")
    ch = Character(rigs["front"], pos=(0.2, 0.95), height=0.7, views=rigs)
    ch.actions = [Action("idle", 0.4), Action("turn", view="three_quarter"), Action("turn", view="side"),
                  Action("walk", target=(0.6, 0.95)), Action("stop"),
                  Action("turn", view="three_quarter_back"), Action("turn", view="back"),
                  Action("idle", 0.5), Action("turn", view="front"), Action("wave")]
    return ch


if __name__ == "__main__":
    s = Scene()
    ch = build_view_character()
    s.characters.append(ch)
    segs = ch.schedule(s.lib)
    print([(round(x.t0, 2), x.clip, x.view, x.from_view) for x in segs])
    times = [0.2, 0.55, 0.75, 1.0, 1.3, 1.7, 2.1, 2.6, 2.95, 3.2, 3.6, 3.9, 4.3, 4.6, 5.0, 5.5]
    sheet(s, times, "test_output/views.png", w=480, h=270, cols=4)
    print("ok", s.duration())
