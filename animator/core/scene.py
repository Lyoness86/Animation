"""Scene = background (image or video) + layered characters. Renders frames
at any size.

Frames are float32 premultiplied BGRA in 0..1."""
import math

import cv2
import numpy as np

from .clips import Library
from .rig import DRAW_ORDER


class BackgroundVideo:
    """Reads frames from a video file for a given time (loops if the scene is
    longer than the video). Sequential reads are fast; jumping around (while
    scrubbing the preview) seeks."""

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        self._next = 0
        self._last = (None, None)
        ok, first = self._read(0)
        if not ok:
            raise ValueError(f"Could not read video: {path}")
        self.size = (first.shape[1], first.shape[0])

    def _read(self, idx):
        if idx != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = self.cap.read()
        if not ok and idx != 0:  # some files report too many frames
            self.count = max(idx, 1)
            return self._read(idx % self.count)
        self._next = idx + 1
        return ok, frame

    def frame_at(self, t):
        idx = int(t * self.fps) % self.count
        if self._last[0] != idx:
            ok, frame = self._read(idx)
            if ok:
                self._last = (idx, frame)
        return self._last[0], self._last[1]

    @property
    def duration(self):
        return self.count / self.fps

    def close(self):
        self.cap.release()


def _cover(img, w, h):
    bh, bw = img.shape[:2]
    s = max(w / bw, h / bh)
    img = cv2.resize(img, (math.ceil(bw * s), math.ceil(bh * s)),
                     interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
    img = img[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255
    return np.ascontiguousarray(np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2))


class Scene:
    def __init__(self, width=1920, height=1080):
        self.width, self.height = width, height
        self.background = None      # BGR uint8 still image
        self.video = None           # BackgroundVideo or None
        self.characters = []        # index 0 = back-most layer
        self.lib = Library()
        self._bg_cache = {}

    def set_background(self, bgr):
        self.clear_background()
        self.background = bgr[..., :3].copy()

    def set_background_video(self, path):
        video = BackgroundVideo(path)
        self.clear_background()
        self.video = video

    def clear_background(self):
        if self.video is not None:
            self.video.close()
        self.background, self.video = None, None
        self._bg_cache.clear()

    def duration(self):
        ends = [c.end_time(self.lib) for c in self.characters]
        return max([3.0] + [e + 1.0 for e in ends])

    # ------------------------------------------------------------ rendering
    def _background(self, w, h, t=0.0):
        if self.video is not None:
            idx, img = self.video.frame_at(t)
            key = ("video", w, h, idx)
            if key not in self._bg_cache:
                self._bg_cache.clear()
                self._bg_cache[key] = _cover(img, w, h)
            return self._bg_cache[key]
        key = (w, h)
        if key not in self._bg_cache:
            if self.background is None:
                g = np.linspace(0.62, 0.42, h, dtype=np.float32)[:, None, None]
                bg = np.concatenate([np.broadcast_to(g, (h, w, 3)), np.ones((h, w, 1), np.float32)], axis=2)
                bg = np.ascontiguousarray(bg)
            else:
                bg = _cover(self.background, w, h)
            self._bg_cache.clear()
            self._bg_cache[key] = bg
        return self._bg_cache[key]

    def render(self, t, w=None, h=None, background=True, only=None):
        """Render the scene at time t. `only` = a character to render alone.
        Returns (frame, boxes) where boxes maps character -> (x0,y0,x1,y1)."""
        w, h = w or self.width, h or self.height
        frame = self._background(w, h, t).copy() if background else np.zeros((h, w, 4), np.float32)
        boxes = {}
        for ch in self.characters:
            if only is not None and ch is not only:
                continue
            boxes[id(ch)] = self.draw_character(frame, ch, t)
        return frame, boxes

    def character_transform(self, ch, pos, facing_left, pose, w, h, rig=None, squeeze=1.0):
        rig = rig or ch.rig
        s = ch.height * h / rig.height
        flip = facing_left != ch.mirrored
        if rig.faces_left:
            flip = not flip
        fx = (-1.0 if flip else 1.0) * squeeze
        feet = rig.J["feet"]
        px = pos[0] * w + pose.get("root.x", 0.0) * ch.height * h
        py = pos[1] * h + pose.get("root.y", 0.0) * ch.height * h
        return np.array([[fx * s, 0, px - fx * s * feet[0]],
                         [0, s, py - s * feet[1]],
                         [0, 0, 1.0]]), s

    def draw_character(self, frame, ch, t):
        h, w = frame.shape[:2]
        segs = ch.schedule(self.lib)
        pose, pos, left = ch.state_at(t, self.lib, segs)
        view, squeeze = ch.view_at(t, self.lib, segs)
        rig = ch.rig_for(view)
        place, s = self.character_transform(ch, pos, left, pose, w, h, rig, squeeze)
        mats = rig.bone_matrices(pose)
        box = [math.inf, math.inf, -math.inf, -math.inf]

        def grow(b):
            if b:
                box[:] = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]

        for kind, item in draw_list(ch.props, pose):
            if kind == "prop":
                grow(draw_prop(frame, item, rig, mats, place, pose))
                continue
            part = rig.parts.get(item)
            if part:
                grow(draw_skinned_part(frame, part, [place @ mats[bn] for bn in part["bones"]], s))
        return tuple(box) if box[0] < math.inf else None


def draw_list(props, pose):
    """Body parts in drawing order with held objects slotted in according to
    their layer (an animation can override it with prop_l/prop_r.layer)."""
    order = [("part", n) for n in DRAW_ORDER]
    for prop in props:
        layer = pose.get(f"prop_{prop.hand}.layer", -1.0)
        layer = prop.layer if layer < -0.5 else int(round(layer))
        fore = ("part", "hand_" + prop.hand)
        if layer <= 0:
            i = order.index(("part", "torso"))
        elif layer == 1:
            i = order.index(fore)
        elif layer == 2:
            i = order.index(fore) + 1
        else:
            i = len(order)
        order.insert(i, ("prop", prop))
    return order


def draw_prop(frame, prop, rig, mats, place, pose):
    """Draw a held object at the palm of its hand. It follows the hand's
    position; its angle stays upright unless prop.follow / prop_x.tilt say
    otherwise."""
    fore = mats["hand_" + prop.hand]
    palm = fore @ np.append(rig.hand_point(prop.hand), 1.0)
    fore_angle = math.degrees(math.atan2(fore[0, 1], fore[0, 0]))
    angle = prop.rotation + pose.get(f"prop_{prop.hand}.tilt", 0.0) + prop.follow * fore_angle
    img = prop.premultiplied()
    ih, iw = img.shape[:2]
    k = prop.size * rig.height / ih
    a = math.radians(angle)
    c, sn = math.cos(a), math.sin(a)
    gx, gy = prop.grip[0] * iw, prop.grip[1] * ih
    cx = palm[0] + prop.offset[0] * rig.height
    cy = palm[1] + prop.offset[1] * rig.height
    P = np.array([[c * k, sn * k, cx - (c * k * gx + sn * k * gy)],
                  [-sn * k, c * k, cy - (-sn * k * gx + c * k * gy)],
                  [0, 0, 1.0]])
    M = place @ P
    # pre-shrink large objects so they don't shimmer when drawn small
    scale = math.hypot(M[0, 0], M[1, 0])
    if scale < 0.5:
        f = max(1, int(1 / scale) // 2 * 2) or 1
        small = cv2.resize(img, (max(1, iw // f), max(1, ih // f)), interpolation=cv2.INTER_AREA)
        M = M @ np.diag([iw / small.shape[1], ih / small.shape[0], 1.0])
        img = small
    return draw_part(frame, img, M[:2])


GRID = 4  # the bending is solved every GRID output pixels, then interpolated


def draw_skinned_part(frame, part, mats, scale):
    """Draw one body part whose pixels follow a weighted blend of bone
    transforms (linear blend skinning). For every output pixel we solve
    "which picture pixel lands here?" with a few Newton steps on a coarse
    grid, then sample the picture with cv2.remap."""
    H, W = frame.shape[:2]
    W0 = part["weights"]
    h0, w0 = W0.shape[:2]
    ox, oy = part["ox"], part["oy"]
    corners = np.array([[ox, ox + w0, ox, ox + w0], [oy, oy, oy + h0, oy + h0], [1, 1, 1, 1]], float)
    pts = np.concatenate([(M @ corners)[:2] for M in mats], axis=1)
    x0 = max(int(math.floor(pts[0].min())) - 1, 0)
    y0 = max(int(math.floor(pts[1].min())) - 1, 0)
    x1 = min(int(math.ceil(pts[0].max())) + 2, W)
    y1 = min(int(math.ceil(pts[1].max())) + 2, H)
    if x1 <= x0 or y1 <= y0:
        return None
    bw, bh = x1 - x0, y1 - y0

    # coarse grid of output positions
    gx = np.arange(0, bw + GRID, GRID, dtype=np.float32) + x0
    gy = np.arange(0, bh + GRID, GRID, dtype=np.float32) + y0
    qx, qy = np.meshgrid(gx, gy)
    A = np.stack([M[:2] for M in mats]).astype(np.float32)  # (K, 2, 3)
    inv0 = np.linalg.inv(mats[0])
    px = inv0[0, 0] * qx + inv0[0, 1] * qy + inv0[0, 2]
    py = inv0[1, 0] * qx + inv0[1, 1] * qy + inv0[1, 2]
    K = len(mats)
    for _ in range(5 if K > 1 else 0):
        wts = _sample_weights(W0, px - ox, py - oy)            # (gh, gw, K)
        L = np.einsum("hwk,kij->hwij", wts, A)                 # blended 2x3 per point
        fx = L[..., 0, 0] * px + L[..., 0, 1] * py + L[..., 0, 2]
        fy = L[..., 1, 0] * px + L[..., 1, 1] * py + L[..., 1, 2]
        rx, ry = qx - fx, qy - fy
        a, b, c, d = L[..., 0, 0], L[..., 0, 1], L[..., 1, 0], L[..., 1, 1]
        det = a * d - b * c
        ok = np.abs(det) > 1e-3 * scale * scale
        det = np.where(ok, det, 1.0)
        px = px + np.where(ok, (d * rx - b * ry) / det, 0)
        py = py + np.where(ok, (-c * rx + a * ry) / det, 0)

    if K > 1:  # where the solve didn't settle (extreme bends)
        wts = _sample_weights(W0, px - ox, py - oy)
        L = np.einsum("hwk,kij->hwij", wts, A)
        fx = L[..., 0, 0] * px + L[..., 0, 1] * py + L[..., 0, 2]
        fy = L[..., 1, 0] * px + L[..., 1, 1] * py + L[..., 1, 2]
        bad = np.hypot(qx - fx, qy - fy) > 4.0
        # fall back to moving those pixels stiffly with their own bone
        # rather than leaving a hole
        rx = inv0[0, 0] * qx + inv0[0, 1] * qy + inv0[0, 2]
        ry = inv0[1, 0] * qx + inv0[1, 1] * qy + inv0[1, 2]
        px = np.where(bad, rx, px)
        py = np.where(bad, ry, py)

    # pick a pre-shrunk copy of the picture close to the output size
    crop, f = part["levels"][0]
    for cand in part["levels"]:
        if cand[1] * scale <= 1.0:
            crop, f = cand
    gridmap = np.dstack([(px - ox) / f, (py - oy) / f]).astype(np.float32)
    ux = (np.arange(bw, dtype=np.float32) / GRID)[None, :].repeat(bh, 0)
    uy = (np.arange(bh, dtype=np.float32) / GRID)[:, None].repeat(bw, 1)
    full = cv2.remap(gridmap, ux, uy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    warped = cv2.remap(crop, full[..., 0], full[..., 1], cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    src = warped.astype(np.float32) * (1.0 / 255)
    dst = frame[y0:y1, x0:x1]
    dst *= 1.0 - src[..., 3:4]
    dst += src
    ys, xs = np.nonzero(warped[..., 3] > 8)
    if not len(xs):
        return None
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def _sample_weights(W0, cx, cy):
    cx = cx.astype(np.float32)
    cy = cy.astype(np.float32)
    chans = []
    for i in range(0, W0.shape[2], 4):
        chunk = np.ascontiguousarray(W0[..., i:i + 4])
        r = cv2.remap(chunk, cx, cy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        chans.append(r if r.ndim == 3 else r[..., None])
    return np.concatenate(chans, axis=2)


def draw_part(frame, crop, M):
    """Warp a premultiplied BGRA uint8 crop with 2x3 matrix M and composite
    it 'over' the float frame. Returns the drawn bbox or None."""
    H, W = frame.shape[:2]
    ch, cw = crop.shape[:2]
    corners = M @ np.array([[0, cw, 0, cw], [0, 0, ch, ch], [1, 1, 1, 1]], float)
    x0 = max(int(math.floor(corners[0].min())) - 1, 0)
    y0 = max(int(math.floor(corners[1].min())) - 1, 0)
    x1 = min(int(math.ceil(corners[0].max())) + 1, W)
    y1 = min(int(math.ceil(corners[1].max())) + 1, H)
    if x1 <= x0 or y1 <= y0:
        return None
    M2 = M.copy()
    M2[0, 2] -= x0
    M2[1, 2] -= y0
    warped = cv2.warpAffine(crop, M2, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    src = warped.astype(np.float32) * (1.0 / 255)
    dst = frame[y0:y1, x0:x1]
    dst *= 1.0 - src[..., 3:4]
    dst += src
    return (x0, y0, x1, y1)


def to_straight_uint8(frame):
    """Premultiplied float BGRA -> straight-alpha uint8 BGRA."""
    a = frame[..., 3:4]
    rgb = np.where(a > 1e-4, frame[..., :3] / np.maximum(a, 1e-4), 0)
    out = np.concatenate([rgb, a], axis=2)
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)


def to_bgr_over(frame, colour=(0, 0, 0)):
    """Premultiplied float BGRA -> opaque BGR uint8 over a solid colour."""
    bg = np.array(colour, np.float32) / 255
    out = frame[..., :3] + bg * (1 - frame[..., 3:4])
    return (np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8)
