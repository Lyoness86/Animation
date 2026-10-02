"""Scene = background + layered characters. Renders frames at any size.

Frames are float32 premultiplied BGRA in 0..1."""
import math

import cv2
import numpy as np

from .clips import Library
from .rig import DRAW_ORDER


class Scene:
    def __init__(self, width=1920, height=1080):
        self.width, self.height = width, height
        self.background = None      # BGR uint8
        self.characters = []        # index 0 = back-most layer
        self.lib = Library()
        self._bg_cache = {}

    def set_background(self, bgr):
        self.background = bgr[..., :3].copy()
        self._bg_cache.clear()

    def duration(self):
        ends = [c.end_time(self.lib) for c in self.characters]
        return max([3.0] + [e + 1.0 for e in ends])

    # ------------------------------------------------------------ rendering
    def _background(self, w, h):
        key = (w, h)
        if key not in self._bg_cache:
            if self.background is None:
                g = np.linspace(0.62, 0.42, h, dtype=np.float32)[:, None, None]
                bg = np.concatenate([np.broadcast_to(g, (h, w, 3)), np.ones((h, w, 1), np.float32)], axis=2)
            else:
                bh, bw = self.background.shape[:2]
                s = max(w / bw, h / bh)  # cover
                img = cv2.resize(self.background, (math.ceil(bw * s), math.ceil(bh * s)), interpolation=cv2.INTER_AREA)
                y0, x0 = (img.shape[0] - h) // 2, (img.shape[1] - w) // 2
                img = img[y0:y0 + h, x0:x0 + w].astype(np.float32) / 255
                bg = np.concatenate([img, np.ones((h, w, 1), np.float32)], axis=2)
            self._bg_cache.clear()
            self._bg_cache[key] = np.ascontiguousarray(bg)
        return self._bg_cache[key]

    def render(self, t, w=None, h=None, background=True, only=None):
        """Render the scene at time t. `only` = a character to render alone.
        Returns (frame, boxes) where boxes maps character -> (x0,y0,x1,y1)."""
        w, h = w or self.width, h or self.height
        frame = self._background(w, h).copy() if background else np.zeros((h, w, 4), np.float32)
        boxes = {}
        for ch in self.characters:
            if only is not None and ch is not only:
                continue
            boxes[id(ch)] = self.draw_character(frame, ch, t)
        return frame, boxes

    def character_transform(self, ch, pos, facing_left, pose, w, h):
        rig = ch.rig
        s = ch.height * h / rig.height
        fx = -1.0 if (facing_left != ch.mirrored) else 1.0
        feet = rig.J["feet"]
        px = pos[0] * w + pose.get("root.x", 0.0) * ch.height * h
        py = pos[1] * h + pose.get("root.y", 0.0) * ch.height * h
        return np.array([[fx * s, 0, px - fx * s * feet[0]],
                         [0, s, py - s * feet[1]],
                         [0, 0, 1.0]]), s

    def draw_character(self, frame, ch, t):
        h, w = frame.shape[:2]
        pose, pos, left = ch.state_at(t, self.lib)
        place, s = self.character_transform(ch, pos, left, pose, w, h)
        mats = ch.rig.bone_matrices(pose)
        box = [math.inf, math.inf, -math.inf, -math.inf]
        for name in DRAW_ORDER:
            part = ch.rig.parts.get(name)
            if not part:
                continue
            b = draw_skinned_part(frame, part, [place @ mats[bn] for bn in part["bones"]], s)
            if b:
                box = [min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3])]
        return tuple(box) if box[0] < math.inf else None


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

    if K > 1:  # where the solve didn't settle (extreme bends) draw nothing
        wts = _sample_weights(W0, px - ox, py - oy)
        L = np.einsum("hwk,kij->hwij", wts, A)
        fx = L[..., 0, 0] * px + L[..., 0, 1] * py + L[..., 0, 2]
        fy = L[..., 1, 0] * px + L[..., 1, 1] * py + L[..., 1, 2]
        bad = np.hypot(qx - fx, qy - fy) > 4.0
        px = np.where(bad, -1e5, px)
        py = np.where(bad, -1e5, py)

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
