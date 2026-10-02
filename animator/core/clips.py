"""Animation clips loaded from JSON files in animator/animations/.

A clip is a set of keyframed channels, e.g. "upper_arm_r.rot" (degrees,
counter-clockwise on screen, relative to the pose in the original picture),
"thigh_l.len" (length scale), "root.y" (vertical offset in character heights,
negative = up), "root.rot", "root.sy" (squash/stretch from the feet).
Clips are authored for a character facing/moving to the right; the renderer
mirrors them for characters facing left.

Each key is [time, value] or [time, value, easing] where easing (to the
next key) is one of: ease (default), linear, in, out, step.

More channels:
  bone.dir / bone.dirw   point a bone in a screen direction (0 = down,
                         90 = right, 180 = up), blended in by dirw (0..1)
  arm_l/arm_r.ikw/ikx/iky  reach the hand to the mouth (+ offset in character
                         heights) with weight ikw - works for any character
  prop_l/prop_r.tilt     extra rotation of an object held in that hand
  prop_l/prop_r.layer    draw order of that object (-1 = its own setting,
                         0 behind body, 1 behind hand, 2 in front of hand,
                         3 in front of everything) - use "step" easing
Clip options: "idle_after": clip to rest with afterwards (e.g. sit_idle),
"hidden": true = no button (used for helper clips).

Adding a new animation = dropping a new .json file in the folder."""
import json
from pathlib import Path

ANIM_DIR = Path(__file__).resolve().parents[1] / "animations"
DEFAULTS = {"len": 1.0, "sy": 1.0, "layer": -1.0}


def default_value(channel):
    return DEFAULTS.get(channel.rsplit(".", 1)[-1], 0.0)


EASINGS = {
    "ease": lambda u: u * u * (3 - 2 * u),   # slow-in/slow-out (default)
    "linear": lambda u: u,
    "out": lambda u: 1 - (1 - u) ** 2,         # fast start, slows down
    "in": lambda u: u * u,                     # slow start, speeds up
    "step": lambda u: 0.0,                     # hold, then jump at the next key
}


class Clip:
    def __init__(self, data, clip_id):
        self.id = clip_id
        self.name = data.get("name", clip_id)
        self.duration = float(data["duration"])
        self.loop = bool(data.get("loop", False))
        # after this clip the character rests with this clip instead of
        # "idle" (e.g. "sit" -> "sit_idle" until "stand")
        self.idle_after = data.get("idle_after")
        self.hidden = bool(data.get("hidden", False))  # internal, no button
        # key = [time, value] or [time, value, easing-to-next-key]
        self.tracks = {ch: sorted((float(k[0]), float(k[1]), k[2] if len(k) > 2 else "ease")
                                  for k in keys)
                       for ch, keys in data["tracks"].items()}
        self.tangents = {ch: _tangents(keys, self.loop, self.duration)
                         for ch, keys in self.tracks.items()}

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
                tans = self.tangents[ch]
                for i, ((t0, v0, e), (t1, v1, _)) in enumerate(zip(keys, keys[1:])):
                    if t0 <= t <= t1:
                        h = t1 - t0
                        u = (t - t0) / h if h > 0 else 1.0
                        if e == "ease":
                            # smooth curve through the keys (no stop at each key)
                            u2, u3 = u * u, u * u * u
                            v = ((2 * u3 - 3 * u2 + 1) * v0 + (u3 - 2 * u2 + u) * h * tans[i]
                                 + (-2 * u3 + 3 * u2) * v1 + (u3 - u2) * h * tans[i + 1])
                        else:
                            v = v0 + (v1 - v0) * EASINGS.get(e, EASINGS["ease"])(u)
                        break
            pose[ch] = v
        return pose


def _tangents(keys, loop, duration):
    """Slopes at each key for a smooth curve that never overshoots (a
    monotone cubic): zero at peaks, holds and the clip's start/end, so values
    like weights never go past their keys."""
    n = len(keys)
    m = [0.0] * n
    if n < 2:
        return m
    t = [k[0] for k in keys]
    v = [k[1] for k in keys]
    for i in range(n):
        if 0 < i < n - 1:
            h0, h1 = t[i] - t[i - 1], t[i + 1] - t[i]
            v_prev = v[i - 1]
        elif loop and n > 2 and abs(t[0]) < 1e-9 and abs(t[-1] - duration) < 1e-9:
            # looping clip: the first and last key are the same moment
            h0, h1 = t[-1] - t[-2], t[1] - t[0]
            v_prev = v[-2]
            i_next = 1
        else:
            continue
        nxt = v[i + 1] if 0 < i < n - 1 else v[i_next]
        if h0 <= 0 or h1 <= 0:
            continue
        d0, d1 = (v[i] - v_prev) / h0, (nxt - v[i]) / h1
        if d0 * d1 > 0:
            m[i] = 3 * (h0 + h1) / ((2 * h1 + h0) / d0 + (h1 + 2 * h0) / d1)
    if loop and n > 2 and abs(t[0]) < 1e-9 and abs(t[-1] - duration) < 1e-9:
        m[-1] = m[0]
    return m


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
