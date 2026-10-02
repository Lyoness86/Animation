"""A character placed in the scene, with its own list of actions.

Actions run one after another from time 0. When the list runs out the
character idles. Positions are normalised scene coordinates (0..1) of the
point between the character's feet."""
import math
from dataclasses import dataclass, field

from .clips import blend

BLEND_TIME = 0.25        # seconds of cross-fade between actions
WALK_SPEED = 0.55        # character heights per second
REF_W, REF_H = 1920, 1080  # reference scene size for speed calculations


@dataclass
class Action:
    kind: str                    # "idle", "walk", or a clip id like "wave"
    duration: float = 0.0        # for "idle"
    target: tuple = None         # for "walk": (x, y) normalised

    def label(self, lib=None):
        if self.kind == "idle":
            return f"Stand still {self.duration:g}s"
        if self.kind == "walk":
            return f"Walk to ({self.target[0]:.2f}, {self.target[1]:.2f})"
        return lib[self.kind].name if lib and self.kind in lib.clips else self.kind.title()


@dataclass
class Segment:
    t0: float
    t1: float
    clip: str
    p0: tuple
    p1: tuple
    facing_left: bool


@dataclass
class Character:
    rig: object
    pos: tuple = (0.5, 0.9)      # start position (feet), normalised
    height: float = 0.55         # fraction of scene height
    mirrored: bool = False       # user flip of the original picture
    actions: list = field(default_factory=list)

    @property
    def name(self):
        return self.rig.name

    def schedule(self, lib):
        segs, t, p, left = [], 0.0, tuple(self.pos), False
        for a in self.actions:
            if a.kind == "idle":
                d = max(a.duration, 0.1)
                segs.append(Segment(t, t + d, "idle", p, p, left))
            elif a.kind == "walk":
                dx = (a.target[0] - p[0]) * REF_W
                dy = (a.target[1] - p[1]) * REF_H
                speed = WALK_SPEED * self.height * REF_H
                d = max(math.hypot(dx, dy) / speed, 0.3)
                if abs(dx) > 1:
                    left = dx < 0
                segs.append(Segment(t, t + d, "walk", p, tuple(a.target), left))
                p = tuple(a.target)
            else:
                if a.kind not in lib.clips:
                    continue
                d = lib[a.kind].duration
                segs.append(Segment(t, t + d, a.kind, p, p, left))
            t = segs[-1].t1
        segs.append(Segment(t, math.inf, "idle", p, p, left))
        return segs

    def end_time(self, lib):
        segs = self.schedule(lib)
        return segs[-1].t0

    def state_at(self, t, lib, segs=None):
        """Returns (pose, (x, y) normalised, facing_left)."""
        segs = segs or self.schedule(lib)
        i = 0
        while i < len(segs) - 1 and t >= segs[i].t1:
            i += 1
        s = segs[i]
        local = max(t - s.t0, 0.0)
        pose = lib[s.clip].sample(local)
        if local < BLEND_TIME and i > 0:
            prev = segs[i - 1]
            prev_pose = lib[prev.clip].sample(prev.t1 - prev.t0)
            u = local / BLEND_TIME
            pose = blend(prev_pose, pose, u * u * (3 - 2 * u))
        if s.p0 != s.p1 and math.isfinite(s.t1):
            u = min(local / (s.t1 - s.t0), 1.0)
            pos = (s.p0[0] + (s.p1[0] - s.p0[0]) * u, s.p0[1] + (s.p1[1] - s.p0[1]) * u)
        else:
            pos = s.p1
        return pose, pos, s.facing_left
