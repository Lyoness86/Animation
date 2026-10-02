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


# ---------------------------------------------------------------- new features
def test_every_library_clip_loads_and_renders():
    scene = Scene()
    ch = Character(_rig())
    scene.characters.append(ch)
    for clip_id, clip in scene.lib.clips.items():
        ch.actions = [Action(clip_id)] if clip_id != "walk" else [Action("walk", target=(0.9, 0.9))]
        for t in (0.0, clip.duration * 0.5):
            frame, boxes = scene.render(t, 320, 180)
            assert np.isfinite(frame).all() and boxes[id(ch)] is not None, clip_id


def test_sit_rests_seated_until_stand():
    scene = Scene()
    ch = Character(_rig())
    ch.actions = [Action("sit"), Action("idle", 1.0), Action("stand")]
    segs = ch.schedule(scene.lib)
    assert [s.clip for s in segs] == ["sit", "sit_idle", "stand", "idle"]


def test_views_switch_during_turn():
    from make_test_character import make_view
    from animator.core.character import VIEW_NAMES
    rigs = {}
    for v in VIEW_NAMES:
        cut = prepare_cutout(make_view(v))
        rigs[v] = Rig(cut, estimate_from_silhouette(cut), v)
    ch = Character(rigs["front"], views=rigs)
    ch.actions = [Action("turn", view="side"), Action("idle", 1.0), Action("turn", view="back")]
    scene = Scene()
    scene.characters.append(ch)
    assert ch.view_at(0.1, scene.lib)[0] == "front"
    assert ch.view_at(0.3, scene.lib)[0] == "side"
    assert ch.view_at(0.2, scene.lib)[1] < 0.5        # squeezed mid-turn
    assert ch.view_at(1.0, scene.lib) == ("side", 1.0)
    assert ch.view_at(5.0, scene.lib) == ("back", 1.0)
    scene.render(1.0, 320, 180)


def test_prop_follows_hand_and_layer_switches():
    from animator.core.character import Prop
    from animator.core.scene import draw_list
    import cv2
    glass = cv2.imread(str(Path(__file__).resolve().parents[1] / "assets/props/glass.png"), cv2.IMREAD_UNCHANGED)
    ch = Character(_rig(), pos=(0.5, 0.95), height=0.8)
    prop = Prop(glass, "Glass", hand="l", layer=1)
    ch.props.append(prop)
    scene = Scene()
    scene.characters.append(ch)
    ch.actions = [Action("drink_other_hand")]
    # held glass is drawn just before the holding hand (fingers over it)...
    pose0, _, _ = ch.state_at(0.0, scene.lib)
    order = [x[1] if x[0] == "part" else "PROP" for x in draw_list(ch.props, pose0)]
    assert order.index("PROP") == order.index("hand_l") - 1
    # ...and in front of everything while drinking
    pose1, _, _ = ch.state_at(1.4, scene.lib)
    assert [x[0] for x in draw_list(ch.props, pose1)][-1] == "prop"
    # the hand (and glass) is much higher at the mouth than at rest
    rig = ch.rig
    hand_rest = rig.bone_matrices(pose0)["forearm_l"] @ np.append(rig.hand_point("l"), 1)
    hand_mouth = rig.bone_matrices(pose1)["forearm_l"] @ np.append(rig.hand_point("l"), 1)
    assert hand_rest[1] - hand_mouth[1] > 0.2 * rig.height
    mouth = rig.mouth_point()
    assert abs(hand_mouth[1] - (mouth[1] + 0.10 * rig.height)) < 0.06 * rig.height


def test_project_round_trip(tmp_path):
    import subprocess
    from animator.core.character import Prop
    from animator.core.export import ffmpeg_exe
    from animator.core.project import load_project, save_project
    video = tmp_path / "bg.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=25:duration=2", "-pix_fmt", "yuv420p", str(video)], check=True)
    scene = Scene()
    scene.set_background_video(video)
    a = Character(_rig(), pos=(0.3, 0.9), height=0.5, mirrored=True)
    a.actions = [Action("walk", target=(0.6, 0.9)), Action("stop"), Action("drink")]
    a.props.append(Prop(np.full((40, 20, 4), 200, np.uint8), "Box", hand="r", size=0.1, layer=2))
    b = Character(_rig(), pos=(0.7, 0.9), height=0.4)
    b.actions = [Action("idle", 0.5), Action("dance")]
    scene.characters += [a, b]
    path = save_project(scene, tmp_path / "p.puppet")
    scene2 = load_project(path)
    assert len(scene2.characters) == 2 and scene2.video is not None
    for t in (0.0, 0.7, 1.9, 3.0):
        f1, _ = scene.render(t, 320, 180)
        f2, _ = scene2.render(t, 320, 180)
        assert np.abs(f1 - f2).max() < 1e-4, t


def test_background_video_follows_time(tmp_path):
    import subprocess
    from animator.core.export import ffmpeg_exe
    video = tmp_path / "bg.mp4"
    subprocess.run([ffmpeg_exe(), "-v", "error", "-y", "-f", "lavfi", "-i",
                    "testsrc2=size=320x180:rate=25:duration=2", "-pix_fmt", "yuv420p", str(video)], check=True)
    scene = Scene()
    scene.set_background_video(video)
    f0, _ = scene.render(0.0, 160, 90)
    f1, _ = scene.render(1.0, 160, 90)
    f0b, _ = scene.render(0.0, 160, 90)          # seeking back works
    assert np.abs(f0 - f1).max() > 0.1
    assert np.abs(f0 - f0b).max() < 1e-6
    scene.render(5.0, 160, 90)                   # loops past the end
