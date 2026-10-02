"""A character placed in the scene, with its own list of actions.

Actions run one after another from time 0. When the list runs out the
character rests (normally "idle"; after e.g. "sit" it rests seated).
Positions are normalised scene coordinates (0..1) of the point between the
character's feet.

A character can have several *views* (front, 3/4, side, 3/4 back, back),
each its own automatically built puppet; a "turn" action switches view.
It can also hold *props* (objects attached to a hand)."""
import math
from dataclasses import dataclass, field

import numpy as np

from .clips import blend

BLEND_TIME = 0.25        # seconds of cross-fade between actions
WALK_SPEED = 0.55        # character heights per second
TURN_TIME = 0.4          # seconds for a turn (view swap at the halfway point)
REF_W, REF_H = 1920, 1080  # reference scene size for speed calculations

VIEW_NAMES = ["front", "three_quarter", "side", "three_quarter_back", "back"]
VIEW_LABELS = {"front": "Front", "three_quarter": "3/4", "side": "Side",
               "three_quarter_back": "3/4 back", "back": "Back"}
PROP_LAYERS = ["Behind body", "Behind hand", "In front of hand", "In front of everything"]


@dataclass
class Action:
    kind: str                    # "idle", "walk", "turn", or a clip id like "wave"
    duration: float = 0.0        # for "idle"
    target: tuple = None         # for "walk": (x, y) normalised
    view: str = None             # for "turn": view name

    def label(self, lib=None):
        if self.kind == "idle":
            return f"Stand still {self.duration:g}s"
        if self.kind == "walk":
            return f"Walk to ({self.target[0]:.2f}, {self.target[1]:.2f})"
        if self.kind == "turn":
            return f"Turn to {VIEW_LABELS.get(self.view, self.view)} view"
        return lib[self.kind].name if lib and self.kind in lib.clips else self.kind.title()


@dataclass
class Prop:
    """An object held in a hand. Hands are named by picture side: "l" = the
    hand on the left of the original picture, "r" = on the right."""
    image: np.ndarray            # BGRA, straight alpha
    name: str = "Object"
    hand: str = "r"
    size: float = 0.15           # object height as a fraction of character height
    offset: tuple = (0.0, 0.0)   # from the palm, in character heights (x right, y down)
    rotation: float = 0.0        # degrees, counter-clockwise
    layer: int = 2               # index into PROP_LAYERS (default: in front of hand)
    grip: tuple = (0.5, 0.6)     # where the hand holds it (fraction of width, height)
    follow: float = 0.0          # 0 = stays upright, 1 = turns fully with the forearm

    def premultiplied(self):
        if getattr(self, "_pm", None) is None or self._pm_src is not self.image:
            a = self.image[..., 3:4].astype(np.float32) / 255
            pm = self.image.astype(np.float32)
            pm[..., :3] *= a
            self._pm = np.ascontiguousarray((pm + 0.5).astype(np.uint8))
            self._pm_src = self.image
        return self._pm


@dataclass
class Segment:
    t0: float
    t1: float
    clip: str
    p0: tuple
    p1: tuple
    facing_left: bool
    view: str = "front"
    from_view: str = None        # set for a turn
    action: int = -1             # index of the step that made this segment


@dataclass
class Character:
    rig: object                  # the front view puppet
    pos: tuple = (0.5, 0.9)      # start position (feet), normalised
    height: float = 0.55         # fraction of scene height
    mirrored: bool = False       # user flip of the original picture
    actions: list = field(default_factory=list)
    views: dict = field(default_factory=dict)   # view name -> Rig (front included)
    start_view: str = "front"
    props: list = field(default_factory=list)
    name: str = ""

    def __post_init__(self):
        self.views.setdefault("front", self.rig)
        if not self.name:
            self.name = self.rig.name

    def rig_for(self, view):
        return self.views.get(view) or self.rig

    def schedule(self, lib):
        segs, t, p, left = [], 0.0, tuple(self.pos), False
        view = self.start_view if self.start_view in self.views else "front"
        rest = "idle"
        for ai, a in enumerate(self.actions):
            n = len(segs)
            if a.kind == "idle":
                d = max(a.duration, 0.1)
                segs.append(Segment(t, t + d, rest, p, p, left, view))
            elif a.kind == "walk" and a.target is not None:
                dx = (a.target[0] - p[0]) * REF_W
                dy = (a.target[1] - p[1]) * REF_H
                speed = WALK_SPEED * self.height * REF_H
                d = max(math.hypot(dx, dy) / speed, 0.3)
                if abs(dx) > 1:
                    left = dx < 0
                segs.append(Segment(t, t + d, "walk", p, tuple(a.target), left, view))
                p = tuple(a.target)
            elif a.kind == "turn":
                if not a.view or a.view not in self.views or a.view == view:
                    continue
                segs.append(Segment(t, t + TURN_TIME, rest, p, p, left, a.view, from_view=view))
                view = a.view
            else:
                if a.kind not in lib.clips or a.kind == "walk":
                    continue
                clip = lib[a.kind]
                segs.append(Segment(t, t + clip.duration, a.kind, p, p, left, view))
                if clip.idle_after and clip.idle_after in lib.clips:
                    rest = clip.idle_after
            if len(segs) > n:
                segs[-1].action = ai
            t = segs[-1].t1 if segs else t
        segs.append(Segment(t, math.inf, rest, p, p, left, view))
        return segs

    def action_times(self, lib):
        """(start, end) for each step, or None for a step that does nothing
        (e.g. turning to the view it is already in)."""
        times = [None] * len(self.actions)
        for sg in self.schedule(lib):
            if sg.action >= 0:
                times[sg.action] = (sg.t0, sg.t1)
        return times

    def action_at(self, t, lib):
        """Index of the step playing at time t, or -1 when resting after the list."""
        segs = self.schedule(lib)
        return segs[self._find(segs, t)].action

    def end_time(self, lib):
        segs = self.schedule(lib)
        return segs[-1].t0

    @staticmethod
    def _find(segs, t):
        i = 0
        while i < len(segs) - 1 and t >= segs[i].t1:
            i += 1
        return i

    def view_at(self, t, lib, segs=None):
        """Returns (view name, horizontal squeeze 0..1) - the squeeze makes a
        turn look like the character spins while the picture is swapped."""
        segs = segs or self.schedule(lib)
        s = segs[self._find(segs, t)]
        if s.from_view is None:
            return s.view, 1.0
        u = min(max((t - s.t0) / (s.t1 - s.t0), 0.0), 1.0)
        return (s.from_view if u < 0.5 else s.view), max(abs(math.cos(math.pi * u)), 0.12)

    def state_at(self, t, lib, segs=None):
        """Returns (pose, (x, y) normalised, facing_left)."""
        segs = segs or self.schedule(lib)
        i = self._find(segs, t)
        s = segs[i]
        rig = self.rig_for(self.view_at(t, lib, segs)[0])
        local = max(t - s.t0, 0.0)
        pose = rig.resolve_pose(lib[s.clip].sample(local))
        if local < BLEND_TIME and i > 0:
            prev = segs[i - 1]
            prev_pose = rig.resolve_pose(lib[prev.clip].sample(prev.t1 - prev.t0))
            u = local / BLEND_TIME
            pose = blend(prev_pose, pose, u * u * (3 - 2 * u))
        if s.p0 != s.p1 and math.isfinite(s.t1):
            u = min(local / (s.t1 - s.t0), 1.0)
            pos = (s.p0[0] + (s.p1[0] - s.p0[0]) * u, s.p0[1] + (s.p1[1] - s.p0[1]) * u)
        else:
            pos = s.p1
        return pose, pos, s.facing_left
