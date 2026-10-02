"""Animation clips loaded from JSON files in animator/animations/.

A clip is a set of keyframed channels, e.g. "upper_arm_r.rot" (degrees,
counter-clockwise on screen, relative to the pose in the original picture),
"thigh_l.len" (length scale), "root.y" (vertical offset in character heights,
negative = up), "root.rot", "root.sy" (squash/stretch from the feet).
Clips are authored for a character facing/moving to the right; the renderer
mirrors them for characters facing left.

Each key is [time, value] or [time, value, easing] where easing (to the
next key) is one of: ease (default), linear, in, out.

Adding a new animation = dropping a new .json file in the folder."""
import json
from pathlib import Path

ANIM_DIR = Path(__file__).resolve().parents[1] / "animations"
DEFAULTS = {"len": 1.0, "sy": 1.0}


def default_value(channel):
    return DEFAULTS.get(channel.rsplit(".", 1)[-1], 0.0)


EASINGS = {
    "ease": lambda u: u * u * (3 - 2 * u),   # slow-in/slow-out (default)
    "linear": lambda u: u,
    "out": lambda u: 1 - (1 - u) ** 2,         # fast start, slows down
    "in": lambda u: u * u,                     # slow start, speeds up
}


class Clip:
    def __init__(self, data, clip_id):
        self.id = clip_id
        self.name = data.get("name", clip_id)
        self.duration = float(data["duration"])
        self.loop = bool(data.get("loop", False))
        # key = [time, value] or [time, value, easing-to-next-key]
        self.tracks = {ch: sorted((float(k[0]), float(k[1]), k[2] if len(k) > 2 else "ease")
                                  for k in keys)
                       for ch, keys in data["tracks"].items()}

    def sample(self, t):
        if self.loop:
            t = t % self.duration
        else:
            t = min(max(t, 0.0), self.duration)
        pose = {}
        for ch, keys in self.tracks.items():
            if t <= keys[0][0]:
                v = keys[0][1]
            elif t >= keys[-1][0]:
                v = keys[-1][1]
            else:
                for (t0, v0, e), (t1, v1, _) in zip(keys, keys[1:]):
                    if t0 <= t <= t1:
                        u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
                        v = v0 + (v1 - v0) * EASINGS.get(e, EASINGS["ease"])(u)
                        break
            pose[ch] = v
        return pose


def blend(a, b, w):
    """Blend pose a -> b by weight w (0..1)."""
    out = {}
    for ch in set(a) | set(b):
        d = default_value(ch)
        out[ch] = a.get(ch, d) * (1 - w) + b.get(ch, d) * w
    return out


class Library:
    def __init__(self, folder=ANIM_DIR):
        self.clips = {}
        for f in sorted(Path(folder).glob("*.json")):
            with open(f, encoding="utf-8") as fh:
                self.clips[f.stem] = Clip(json.load(fh), f.stem)

    def __getitem__(self, k):
        return self.clips[k]
