"""Finds the character's joints automatically.

First tries MediaPipe pose detection (AI). If that fails or looks wrong, falls
back to an estimate based on typical cartoon body proportions. Joint names use
*image* left/right ("_l" = left side of the picture)."""
import os
import urllib.request
from pathlib import Path

import cv2
import numpy as np

JOINTS = ["head_top", "neck",
          "shoulder_l", "shoulder_r", "elbow_l", "elbow_r", "wrist_l", "wrist_r",
          "hip_l", "hip_r", "knee_l", "knee_r", "ankle_l", "ankle_r",
          "hand_l", "hand_r", "toe_l", "toe_r"]  # hand = fingertips, toe = tip of the foot

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/latest/pose_landmarker_full.task")
MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "pose_landmarker_full.task"


def ensure_model():
    """Download the pose model on first use. Returns path or None."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        return MODEL_PATH
    try:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        tmp = MODEL_PATH.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=60) as r, open(tmp, "wb") as f:
            f.write(r.read())
        os.replace(tmp, MODEL_PATH)
        return MODEL_PATH
    except Exception as e:  # offline etc. - fall back to estimation
        print("Pose model download failed:", e)
        return None


def _silhouette_info(alpha):
    ys, xs = np.nonzero(alpha > 128)
    return xs.min(), ys.min(), xs.max(), ys.max(), xs, ys


def _top_in_band(alpha, x, half_width, default_y):
    x0, x1 = int(max(x - half_width, 0)), int(min(x + half_width + 1, alpha.shape[1]))
    rows = np.nonzero((alpha[:, x0:x1] > 128).any(axis=1))[0]
    return float(rows.min()) if len(rows) else default_y


def detect_mediapipe(rgba):
    """Returns joints dict or None."""
    path = ensure_model()
    if path is None:
        return None
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
    except Exception as e:
        print("MediaPipe unavailable:", e)
        return None

    h, w = rgba.shape[:2]
    pad = int(0.15 * h)
    a = rgba[..., 3:4].astype(np.float32) / 255
    rgb = (rgba[..., 2::-1].astype(np.float32) * a + 128 * (1 - a)).astype(np.uint8)
    canvas = np.full((h + 2 * pad, w + 2 * pad, 3), 128, np.uint8)
    canvas[pad:pad + h, pad:pad + w] = rgb

    opts = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(path)),
        running_mode=vision.RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
    try:
        with vision.PoseLandmarker.create_from_options(opts) as lm:
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(canvas)))
    except Exception as e:
        print("Pose detection failed:", e)
        return None
    if not res.pose_landmarks:
        return None
    L = res.pose_landmarks[0]
    ch, cw = canvas.shape[:2]

    def p(i):
        return np.array([L[i].x * cw - pad, L[i].y * ch - pad])

    vis = np.mean([L[i].visibility for i in (11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28)])
    if vis < 0.35:
        return None

    j = {}
    # MediaPipe pairs (person-left, person-right); assign by picture side.
    for name, (i1, i2) in {"shoulder": (11, 12), "elbow": (13, 14), "wrist": (15, 16),
                           "hip": (23, 24), "knee": (25, 26), "ankle": (27, 28)}.items():
        a1, a2 = p(i1), p(i2)
        j[name + "_l"], j[name + "_r"] = (a1, a2) if a1[0] <= a2[0] else (a2, a1)
    # fingertips and toes, attached to whichever wrist/ankle they belong to
    alpha = rgba[..., 3]
    for wi, (f1, f2) in ((15, (17, 19)), (16, (18, 20))):
        w = p(wi)
        side = "l" if np.allclose(j["wrist_l"], w) else "r"
        j["hand_" + side] = _onto_body(alpha, w, w + 1.6 * ((p(f1) + p(f2)) / 2 - w))
    for ai, ti in ((27, 31), (28, 32)):
        a = p(ai)
        side = "l" if np.allclose(j["ankle_l"], a) else "r"
        j["toe_" + side] = _onto_body(alpha, a, p(ti))
    nose = p(0)
    mid_sh = (j["shoulder_l"] + j["shoulder_r"]) / 2
    j["neck"] = mid_sh + 0.25 * (nose - mid_sh)
    sw = np.linalg.norm(j["shoulder_l"] - j["shoulder_r"])
    j["head_top"] = np.array([nose[0], _top_in_band(rgba[..., 3], nose[0], sw * 0.3, nose[1] - sw)])
    j = {k: (float(v[0]), float(v[1])) for k, v in j.items()}
    return j if sanity_check(j, rgba) else None


def _onto_body(alpha, base, tip):
    """Pull a fingertip/toe point back towards its wrist/ankle until it lies
    on the character (a hand in a pocket gets a short hand)."""
    h, w = alpha.shape
    for u in np.linspace(1.0, 0.0, 21):
        q = base + u * (tip - base)
        x, y = int(round(q[0])), int(round(q[1]))
        if 0 <= x < w and 0 <= y < h and alpha[y, x] > 128:
            return q
    return base + 0.3 * (tip - base)


def sanity_check(j, rgba):
    """Reject obviously broken detections."""
    h, w = rgba.shape[:2]
    for x, y in j.values():
        if not (-0.1 * w <= x <= 1.1 * w and -0.1 * h <= y <= 1.1 * h):
            return False
    y = lambda k: (j[k + "_l"][1] + j[k + "_r"][1]) / 2
    return j["head_top"][1] < j["neck"][1] < y("shoulder") + 5 < y("hip") < y("knee") < y("ankle")


def estimate_from_silhouette(rgba):
    """Fallback: place joints using typical cartoon proportions."""
    alpha = rgba[..., 3]
    x0, y0, x1, y1, xs, ys = _silhouette_info(alpha)
    H = y1 - y0
    cx = float(np.median(xs))

    # Neck = narrowest row in the upper part of the body.
    widths = []
    for yy in range(int(y0 + 0.15 * H), int(y0 + 0.45 * H)):
        row = np.nonzero(alpha[yy] > 128)[0]
        widths.append((row.max() - row.min()) if len(row) else 1e9)
    neck_y = y0 + 0.15 * H + int(np.argmin(widths)) if widths else y0 + 0.3 * H
    body = H - (neck_y - y0)
    sh_y = neck_y + 0.06 * body
    hip_y = neck_y + 0.42 * body
    ground = y1
    knee_y = hip_y + 0.5 * (ground - hip_y)
    ankle_y = ground - 0.05 * H

    # Torso half-width: silhouette width at chest, excluding arms (estimate).
    sh_dx = 0.16 * H
    hip_dx = 0.08 * H
    j = {
        "head_top": (cx, float(y0)), "neck": (cx, float(neck_y)),
        "shoulder_l": (cx - sh_dx, sh_y), "shoulder_r": (cx + sh_dx, sh_y),
        "elbow_l": (cx - sh_dx * 1.15, sh_y + 0.45 * (hip_y - sh_y)),
        "elbow_r": (cx + sh_dx * 1.15, sh_y + 0.45 * (hip_y - sh_y)),
        "wrist_l": (cx - sh_dx * 1.2, hip_y), "wrist_r": (cx + sh_dx * 1.2, hip_y),
        "hip_l": (cx - hip_dx, hip_y), "hip_r": (cx + hip_dx, hip_y),
        "knee_l": (cx - hip_dx, knee_y), "knee_r": (cx + hip_dx, knee_y),
        "ankle_l": (cx - hip_dx, ankle_y), "ankle_r": (cx + hip_dx, ankle_y),
    }
    for s in "lr":
        w, e = np.array(j["wrist_" + s]), np.array(j["elbow_" + s])
        j["hand_" + s] = tuple(w + 0.35 * (w - e))
        j["toe_" + s] = (j["ankle_" + s][0], float(y1))
    return {k: (float(v[0]), float(v[1])) for k, v in j.items()}


def plausible(j, rgba):
    """Stricter check of an AI result: legs must stand on the ground and the
    body parts must be at believable heights (catches the AI seeing a side
    view as two overlapping people)."""
    ys = np.nonzero(rgba[..., 3] > 128)[0]
    top, H = ys.min(), ys.max() - ys.min()
    f = lambda k: (j[k][1] - top) / H
    try:
        return (all(f("ankle_" + s) > 0.75 for s in "lr")
                and all(0.3 < f("hip_" + s) < 0.75 for s in "lr")
                and all(f("hip_" + s) < f("knee_" + s) < f("ankle_" + s) for s in "lr")
                and all(0.1 < f("shoulder_" + s) < 0.55 for s in "lr"))
    except KeyError:
        return False


def estimate_from_reference(rgba, ref_joints, ref_rgba, view):
    """Joints for a side / back picture, using the front view's (checked)
    dots: the same heights, placed on this picture's body centre line (hair
    ignored). AI pose detection is unreliable on cartoon side/back views."""
    alpha = rgba[..., 3] > 128
    h, w = alpha.shape
    ys, xs = np.nonzero(alpha)
    top, H = ys.min(), ys.max() - ys.min()
    ref_alpha = ref_rgba[..., 3]
    rys = np.nonzero(ref_alpha > 128)[0]
    rtop, rH = rys.min(), rys.max() - rys.min()
    ref = {k: np.array(v, float) for k, v in ref_joints.items()}
    rcx = (ref["hip_l"][0] + ref["hip_r"][0]) / 2
    fy = {k: (v[1] - rtop) / rH for k, v in ref.items()}

    # body = character minus hair (hair colour sampled at the top of the head)
    lab = cv2.cvtColor(np.ascontiguousarray(rgba[..., :3]), cv2.COLOR_BGR2Lab).astype(np.float32)
    band = alpha[top:top + max(int(0.06 * H), 2)]
    by, bx = np.nonzero(band)
    hair_col = np.median(lab[top + by, bx], axis=0)
    hair = (np.sqrt((((lab - hair_col) * np.array([0.3, 1, 1], np.float32)) ** 2).sum(2)) < 30) & alpha
    body = alpha & ~hair

    def centre(y, wide=0.015):
        y0, y1 = int(max(y - wide * H, 0)), int(min(y + wide * H + 1, h))
        cols = np.nonzero(body[y0:y1].any(axis=0))[0]
        if len(cols) < 3:
            cols = np.nonzero(alpha[y0:y1].any(axis=0))[0]
        return float(np.median(cols)) if len(cols) else float(np.median(xs))

    j = {}
    if view == "side":
        for k in ("shoulder", "elbow", "wrist", "hip", "knee", "ankle"):
            for s, dx in (("l", -2.0), ("r", 2.0)):
                y = top + fy[f"{k}_{s}"] * H
                j[f"{k}_{s}"] = (centre(y) + dx, y)
        # the arm overlaps the body in profile: find it by the skin colour of
        # the front view's hands (forearm/hand rows have no other skin)
        rlab = cv2.cvtColor(np.ascontiguousarray(ref_rgba[..., :3]), cv2.COLOR_BGR2Lab).astype(np.float32)
        samples = []
        for s in "lr":
            wr = ref["wrist_" + s]
            tip = ref.get("hand_" + s, wr + 0.35 * (wr - ref["elbow_" + s]))
            mx, my = (wr + tip) / 2
            r = max(int(0.012 * rH), 2)
            patch = rlab[int(my) - r:int(my) + r + 1, int(mx) - r:int(mx) + r + 1]
            pa = ref_alpha[int(my) - r:int(my) + r + 1, int(mx) - r:int(mx) + r + 1] > 200
            if pa.any():
                samples.append(patch[pa])
        if samples:
            skin_col = np.median(np.concatenate(samples), axis=0)
            skin = (np.sqrt((((lab - skin_col) * np.array([0.4, 1, 1], np.float32)) ** 2).sum(2)) < 20) & body

            def arm_x(y):
                y0, y1 = int(max(y - 0.02 * H, 0)), int(min(y + 0.02 * H + 1, h))
                cols = np.nonzero(skin[y0:y1].any(axis=0))[0]
                return float(np.median(cols)) if len(cols) >= 4 else None

            ex = arm_x(top + fy["elbow_r"] * H)
            wx = arm_x(top + fy["wrist_r"] * H)
            if ex is not None and wx is not None:
                for s, dx in (("l", -2.0), ("r", 2.0)):
                    j["elbow_" + s] = (ex + dx, j["elbow_" + s][1])
                    j["wrist_" + s] = (wx + dx, j["wrist_" + s][1])
                    j["shoulder_" + s] = (ex + dx, j["shoulder_" + s][1])
        # feet point the way the character faces: the side where the shoe sticks out
        gy = int(top + 0.97 * H)
        cols = np.nonzero(alpha[gy:int(top + H) + 1].any(axis=0))[0]
        ax = (j["ankle_l"][0] + j["ankle_r"][0]) / 2
        forward = -1 if len(cols) and (ax - cols.min()) > (cols.max() - ax) else 1
        reach = (ax - cols.min()) if forward < 0 else (cols.max() - ax)
        for s in "lr":
            j["toe_" + s] = (ax + forward * 0.8 * reach, top + 0.98 * H)
            j["hand_" + s] = (j["wrist_" + s][0], j["wrist_" + s][1] + 0.06 * H)
    else:  # back views: the front layout, mirrored, scaled to this picture
        cx = centre(top + fy["hip_l"] * H, wide=0.04)
        for k, v in ref.items():
            if k in ("neck", "head_top"):
                continue
            mk = k[:-1] + ("r" if k.endswith("l") else "l") if k[-2:] in ("_l", "_r") else k
            j[mk] = (cx - (v[0] - rcx) * H / rH, top + fy[k] * H)
    sh = (np.array(j["shoulder_l"]) + np.array(j["shoulder_r"])) / 2
    j["neck"] = (sh[0], top + fy["neck"] * H)
    j["head_top"] = (centre(top + 0.02 * H), float(top))
    for k in ("hand_l", "hand_r", "toe_l", "toe_r"):
        base = np.array(j[("wrist_" if k.startswith("hand") else "ankle_") + k[-1]])
        j[k] = tuple(_onto_body(rgba[..., 3], base, np.array(j[k])))
    return {k: (float(v[0]), float(v[1])) for k, v in j.items()}


def detect_joints(rgba, reference=None, view="front"):
    """Returns (joints, method) where method is 'ai', 'from_front' or
    'estimated'. reference = (front joints, front BGRA picture) for extra views."""
    if reference is not None and view in ("side", "back", "three_quarter_back"):
        return estimate_from_reference(rgba, reference[0], reference[1], view), "from_front"
    j = detect_mediapipe(rgba)
    if j is not None and plausible(j, rgba):
        return j, "ai"
    if reference is not None:
        return estimate_from_reference(rgba, reference[0], reference[1], "back"
                                       if view == "back" else "side"), "from_front"
    return estimate_from_silhouette(rgba), "estimated"
