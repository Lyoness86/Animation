"""Headless checks of the animation engine (run: python -m pytest tests)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from animator.core.character import Action, Character  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import BONE_NAMES, Rig  # noqa: E402
from animator.core.scene import Scene  # noqa: E402
from animator.core.skeleton import estimate_from_silhouette  # noqa: E402
from make_test_character import make_character  # noqa: E402


def _rig():
    cut = prepare_cutout(make_character())
    return Rig(cut, estimate_from_silhouette(cut), "Test")


def test_keying_makes_background_transparent():
    cut = prepare_cutout(make_character())
    assert cut.shape[2] == 4
    assert cut[0, 0, 3] == 0                       # corner is background
    assert cut[cut.shape[0] // 2, cut.shape[1] // 2, 3] == 255  # body is solid


def test_rig_has_all_parts():
    rig = _rig()
    assert set(rig.parts) == set(BONE_NAMES)


def test_walk_reaches_target_and_actions_run_in_order():
    scene = Scene()
    ch = Character(_rig(), pos=(0.2, 0.9))
    ch.actions = [Action("walk", target=(0.6, 0.9)), Action("stop"), Action("wave")]
    scene.characters.append(ch)
    segs = ch.schedule(scene.lib)
    assert [s.clip for s in segs] == ["walk", "stop", "wave", "idle"]
    _, pos, _ = ch.state_at(segs[1].t0 + 0.01, scene.lib)
    assert abs(pos[0] - 0.6) < 1e-6
    frame, boxes = scene.render(0.5, 640, 360)
    assert frame.shape == (360, 640, 4) and boxes[id(ch)] is not None


def test_transparent_render_has_alpha():
    scene = Scene()
    scene.characters.append(Character(_rig()))
    frame, _ = scene.render(0.0, 320, 180, background=False)
    a = frame[..., 3]
    assert a.min() == 0 and a.max() > 0.99
    assert np.isfinite(frame).all()


def test_keying_works_on_yellow_and_green_backgrounds():
    for bg in [(60, 245, 235), (40, 200, 40)]:  # BGR yellow, green
        cut = prepare_cutout(make_character(bg=bg))
        assert cut[0, 0, 3] == 0
        assert cut[cut.shape[0] // 2, cut.shape[1] // 2, 3] == 255
        # no background colour left on solid pixels near the edge
        assert (cut[..., 3] > 0).mean() < 0.75
