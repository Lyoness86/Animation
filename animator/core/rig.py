"""Automatic puppet rig: splits a cut-out into body parts using the detected
joints, fills in hidden areas behind the arms, and computes part transforms
for a pose."""
import math

import cv2
import numpy as np

# name, parent, start joint (pivot), end joint, thickness (x shoulder width)
BONES = [
    ("torso", None, "pelvis", "neck", 0.50),
    ("head", "torso", "neck", "head_top", None),  # thickness from head length
    ("upper_arm_l", "torso", "shoulder_l", "elbow_l", 0.20),
    ("forearm_l", "upper_arm_l", "elbow_l", "wrist_l", 0.20),
    ("hand_l", "forearm_l", "wrist_l", "hand_l", 0.17),
    ("upper_arm_r", "torso", "shoulder_r", "elbow_r", 0.20),
    ("forearm_r", "upper_arm_r", "elbow_r", "wrist_r", 0.20),
    ("hand_r", "forearm_r", "wrist_r", "hand_r", 0.17),
    ("thigh_l", None, "hip_l", "knee_l", 0.26),
    ("shin_l", "thigh_l", "knee_l", "ankle_l", 0.21),
    ("foot_l", "shin_l", "ankle_l", "toe_l", 0.17),
    ("thigh_r", None, "hip_r", "knee_r", 0.26),
    ("shin_r", "thigh_r", "knee_r", "ankle_r", 0.21),
    ("foot_r", "shin_r", "ankle_r", "toe_r", 0.17),
]
BONE_NAMES = [b[0] for b in BONES]
DRAW_ORDER = ["thigh_l", "thigh_r", "shin_l", "shin_r", "foot_l", "foot_r", "torso", "head",
              "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r", "hand_l", "hand_r"]
# Smooth bending: near each joint the picture blends between the two bones
# (parent, child, joint, blend radius as a fraction of shoulder width).
JOINT_BLENDS = [
    ("torso", "head", "neck", 0.22),
    ("torso", "upper_arm_l", "shoulder_l", 0.18), ("upper_arm_l", "forearm_l", "elbow_l", 0.16),
    ("torso", "upper_arm_r", "shoulder_r", 0.18), ("upper_arm_r", "forearm_r", "elbow_r", 0.16),
    ("torso", "thigh_l", "hip_l", 0.22), ("thigh_l", "shin_l", "knee_l", 0.17),
    ("torso", "thigh_r", "hip_r", 0.22), ("thigh_r", "shin_r", "knee_r", 0.17),
    ("forearm_l", "hand_l", "wrist_l", 0.08), ("forearm_r", "hand_r", "wrist_r", 0.08),
    ("shin_l", "foot_l", "ankle_l", 0.08), ("shin_r", "foot_r", "ankle_r", 0.08),
]
BONE_PARENT = {b[0]: b[1] for b in BONES}

PART_COLOURS = {  # for the "check the dots" overlay (BGR)
    "torso": (60, 180, 75), "head": (25, 225, 255), "upper_arm_l": (200, 130, 0),
    "forearm_l": (240, 50, 230), "upper_arm_r": (48, 130, 245), "forearm_r": (180, 30, 145),
    "thigh_l": (75, 25, 230), "shin_l": (128, 128, 0), "thigh_r": (0, 128, 128), "shin_r": (195, 255, 170),
    "hand_l": (255, 190, 220), "hand_r": (0, 215, 255), "foot_l": (180, 105, 255), "foot_r": (40, 40, 160),
}


def derived_joints(joints, alpha):
    """Adds pelvis and ground point to the user-facing joints, plus finger
    and toe tips when an older project doesn't have them."""
    j = {k: np.array(v, float) for k, v in joints.items()}
    j["pelvis"] = (j["hip_l"] + j["hip_r"]) / 2
    ys, xs = np.nonzero(alpha > 128)
    ground = float(ys.max())
    for s in "lr":
        if "hand_" + s not in j:
            j["hand_" + s] = j["wrist_" + s] + 0.35 * (j["wrist_" + s] - j["elbow_" + s])
        if "toe_" + s not in j:
            j["toe_" + s] = np.array([j["ankle_" + s][0], max(ground, j["ankle_" + s][1] + 1)])
    j["feet"] = np.array([j["pelvis"][0], ground])
    return j


END_JOINTS = {"pelvis", "head_top", "hand_l", "hand_r", "toe_l", "toe_r"}


def _seg_dist(px, py, a, b, pen_a=0.0, pen_b=0.0):
    """Distance to segment a-b. Pixels beyond an end get an extra penalty so
    the neighbouring bone claims them (e.g. the chin belongs to the head)."""
    d = b - a
    L2 = max(float(d @ d), 1e-6)
    t_raw = ((px - a[0]) * d[0] + (py - a[1]) * d[1]) / L2
    t = np.clip(t_raw, 0, 1)
    dist = np.hypot(px - (a[0] + t * d[0]), py - (a[1] + t * d[1]))
    L = np.sqrt(L2)
    over = np.where(t_raw < 0, -t_raw * pen_a, 0) + np.where(t_raw > 1, (t_raw - 1) * pen_b, 0)
    return dist + over * L


def _geodesic(mask, seeds, max_steps):
    """Approximate distance (in pixels) from seeds, travelling only inside
    mask. Unreached pixels are inf."""
    dist = np.full(mask.shape, np.inf, np.float32)
    reached = (seeds & mask).astype(np.uint8)
    if not reached.any():
        return dist
    dist[reached > 0] = 0
    m8 = mask.astype(np.uint8)
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    square = np.ones((3, 3), np.uint8)
    for step in range(1, max_steps + 1):
        grown = cv2.dilate(reached, square if step % 2 else cross) & m8
        new = (grown > 0) & (reached == 0)
        if not new.any():
            break
        dist[new] = step * 1.08  # octagonal approximation of true distance
        reached = grown
    return dist


def _premultiply(bgra):
    a = bgra[..., 3:4].astype(np.float32) / 255
    out = bgra.astype(np.float32)
    out[..., :3] *= a
    return (out + 0.5).astype(np.uint8)


class Rig:
    """A ready-to-animate character."""

    def __init__(self, rgba, joints, name="Character", faces_left=False):
        self.name = name
        self.faces_left = faces_left  # picture shows the character facing left
        self.image = rgba
        self.joints = {k: tuple(v) for k, v in joints.items()}
        self.build()

    # ------------------------------------------------------------------ build
    def build(self):
        alpha = self.image[..., 3]
        J = derived_joints(self.joints, alpha)
        self.J = J
        sw = max(np.linalg.norm(J["shoulder_l"] - J["shoulder_r"]), 10.0)
        head_len = np.linalg.norm(J["head_top"] - J["neck"])
        self.height = float(J["feet"][1] - J["head_top"][1])

        h, w = alpha.shape
        solid = alpha > 0
        labels = self._segment(J, solid, sw, head_len)
        labels = self._clean_labels(labels, alpha)
        self.labels = labels
        idx = {n: i for i, n in enumerate(BONE_NAMES)}

        # The torso fills its whole outline (convex hull), so limbs moving
        # away never leave holes. Arm pixels inside it are painted over
        # (inpainted) to invent the shirt hidden behind the arm.
        ty, tx = np.nonzero(labels == idx["torso"])
        hull = np.zeros((h, w), np.uint8)
        if len(tx):
            # include the shoulder and hip joints so the shoulders exist
            # behind the sleeves when an arm is raised
            up = np.array([0.0, -0.12 * sw])
            out_l = (J["shoulder_l"] - J["neck"]) * 0.15
            out_r = (J["shoulder_r"] - J["neck"]) * 0.15
            extra = np.array([J["shoulder_l"] + up + out_l, J["shoulder_r"] + up + out_r,
                              J["shoulder_l"], J["shoulder_r"], J["hip_l"], J["hip_r"], J["neck"]])
            # bodies are roughly symmetric: mirror the visible torso across
            # its centre line so a side hidden behind an arm is filled too
            a, b = J["pelvis"], J["neck"]
            d = (b - a) / max(np.linalg.norm(b - a), 1e-6)
            sh_y = (J["shoulder_l"][1] + J["shoulder_r"][1]) / 2
            hip_y = (J["hip_l"][1] + J["hip_r"][1]) / 2
            chest = ty < sh_y + 0.6 * (hip_y - sh_y)  # arms mostly hide the chest sides
            tp = np.stack([tx[chest], ty[chest]], 1).astype(np.float64)[::7]
            rel = tp - a
            mirrored = a + 2 * np.outer(rel @ d, d) - rel
            pts = np.concatenate([np.stack([tx, ty], 1), mirrored, extra]).astype(np.int32)
            cv2.fillConvexPoly(hull, cv2.convexHull(pts), 1)
        hull = hull.astype(bool) & solid
        arms = np.isin(labels, [idx[n] for n in ("upper_arm_l", "forearm_l", "hand_l",
                                                 "upper_arm_r", "forearm_r", "hand_r")])
        # also fill the body where a hand rests on the hips/legs
        # (only where the hand is surrounded by body - closing fills small
        # gaps enclosed by the body, not the space beside a hanging hand)
        body = np.isin(labels, [idx[n] for n in ("torso", "thigh_l", "thigh_r")]).astype(np.uint8)
        k = max(3, int(0.3 * sw) | 1)
        enclosed = cv2.morphologyEx(body, cv2.MORPH_CLOSE,
                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))).astype(bool)
        hands = np.isin(labels, [idx["hand_l"], idx["hand_r"]])
        behind = (hull & arms) | (enclosed & hands & solid)
        hull = hull | behind
        torso_img = self.image.copy()
        if behind.any():
            # fill from shirt pixels only (never from the arm's own skin)
            # (transparent pixels still hold the old background colour, so
            # they are excluded as sources too)
            fill_mask = cv2.dilate((arms & solid).astype(np.uint8) * 255, np.ones((5, 5), np.uint8))
            fill_mask[alpha < 128] = 255
            by, bx = np.nonzero(hull)
            m = int(0.1 * sw)
            y0, y1 = max(by.min() - m, 0), min(by.max() + m + 1, h)
            x0, x1 = max(bx.min() - m, 0), min(bx.max() + m + 1, w)
            filled = cv2.inpaint(np.ascontiguousarray(self.image[y0:y1, x0:x1, :3]),
                                 np.ascontiguousarray(fill_mask[y0:y1, x0:x1]), 7, cv2.INPAINT_TELEA)
            torso_img[y0:y1, x0:x1, :3] = np.where(behind[y0:y1, x0:x1, None], filled,
                                                   torso_img[y0:y1, x0:x1, :3])
            torso_img[behind, 3] = 255

        # Part masks, with the parent duplicating child pixels near each joint
        # so that rotating a limb never opens a hole.
        masks = {n: (labels == i) for i, n in enumerate(BONE_NAMES)}
        masks["torso"] = masks["torso"] | hull
        for n in ("thigh_l", "thigh_r"):  # torso is drawn over the hips anyway
            masks[n] = masks[n] & ~hull
        yy, xx = np.mgrid[0:h, 0:w]
        for name, parent, j0, j1, th in BONES:
            host = parent or ("torso" if name.startswith("thigh") else None)
            if host is None:
                continue
            thick = 0.42 * head_len if th is None else th * sw
            r = 0.7 * thick
            near = ((xx - J[j0][0]) ** 2 + (yy - J[j0][1]) ** 2) < r * r
            masks[host] = masks[host] | (near & masks[name])

        self.parts = {}
        for name in BONE_NAMES:
            src = torso_img if name == "torso" else self.image
            m = masks[name] & (alpha > 0)
            if not m.any():
                continue
            py, px = np.nonzero(m)
            y0, y1, x0, x1 = py.min(), py.max() + 1, px.min(), px.max() + 1
            crop = src[y0:y1, x0:x1].copy()
            crop[..., 3] = np.where(m[y0:y1, x0:x1], crop[..., 3], 0)
            crop = cv2.copyMakeBorder(_premultiply(crop), 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
            ox, oy = float(x0 - 1), float(y0 - 1)
            bones, weights = self._skin_weights(name, crop.shape[:2], ox, oy, sw)
            levels = [(crop, 1.0)]
            f = 1.0
            while min(crop.shape[:2]) > 8 and f < 8:
                crop = cv2.resize(crop, (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)),
                                  interpolation=cv2.INTER_AREA)
                f *= 2
                levels.append((crop, f))
            self.parts[name] = {"levels": levels, "ox": ox, "oy": oy,
                                "bones": bones, "weights": weights}

        # rest direction of every bone (0 = pointing down, counter-clockwise +)
        self.rest_angle = {}
        for name, parent, j0, j1, _ in BONES:
            d = J[j1] - J[j0]
            self.rest_angle[name] = math.degrees(math.atan2(d[0], d[1]))

    def _skin_weights(self, part, shape, ox, oy, sw):
        """Per-pixel bone weights for one part (own bone first). Near a joint
        the weight moves smoothly towards the neighbouring bone, so the
        picture bends there instead of tearing."""
        h, w = shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        xx += ox
        yy += oy
        bones = [part]
        W = [np.ones((h, w), np.float32)]
        for P, C, jn, rf in JOINT_BLENDS:
            if part not in (P, C):
                continue
            if part == "torso" and P == "torso":
                continue  # torso stays solid; limbs are drawn over the joint
            other = C if part == P else P
            j = self.J[jn]
            r = rf * sw
            R = 2.2 * r
            child_dir = self.J[[b for b in BONES if b[0] == C][0][3]] - j
            n = np.linalg.norm(child_dir)
            child_dir = child_dir / n if n > 1e-6 else np.array([0.0, 1.0])
            s_ = (xx - j[0]) * child_dir[0] + (yy - j[1]) * child_dir[1]
            if P == "torso":
                # big joints: the torso stays solid, only the limb/head bends
                t = np.clip(s_ / (2 * r), 0, 1)
            else:
                t = np.clip((s_ + r) / (2 * r), 0, 1)
            t = t * t * (3 - 2 * t)
            d = np.hypot(xx - j[0], yy - j[1])
            g = np.clip((R - d) / (0.5 * R), 0, 1)
            g = g * g * (3 - 2 * g)
            amt = (t if part == P else 1 - t) * g
            if amt.max() < 1e-3:
                continue
            bones.append(other)
            W.append(amt.astype(np.float32))
            W[0] = W[0] - amt
        W = np.stack(W, axis=2)
        W[..., 0] = np.maximum(W[..., 0], 0)
        W /= np.maximum(W.sum(axis=2, keepdims=True), 1e-6)
        return bones, np.ascontiguousarray(W, dtype=np.float32)

    def hand_point(self, side):
        """Rest-picture point where a held object sits (palm), side l/r."""
        w = self.J["wrist_" + side]
        return w + 0.45 * (self.J["hand_" + side] - w)

    def mouth_point(self):
        return self.J["neck"] + 0.3 * (self.J["head_top"] - self.J["neck"])

    def resolve_pose(self, pose):
        """Turn the 'helper' channels into ordinary rotations:
        - bone.dir/bone.dirw: point a bone in a screen direction (0 = down,
          90 = right, 180 = up), so e.g. a wave looks the same whatever pose
          the arm was drawn in;
        - arm_l/arm_r.ikw (+ikx, iky): reach that hand to the mouth (plus an
          offset in character heights) - two-bone inverse kinematics."""
        helpers = (".dir", ".dirw", ".ikw", ".ikx", ".iky")
        if not any(k.endswith(helpers) for k in pose):
            return pose
        out = dict(pose)
        accum = {}
        for name, parent, *_ in BONES:
            base = accum[parent] if parent else out.get("root.rot", 0.0)
            rot = out.get(name + ".rot", 0.0)
            w = out.get(name + ".dirw", 0.0)
            if w > 0 and name + ".dir" in out:
                target = out[name + ".dir"] - self.rest_angle[name] - base
                rot = rot + w * _wrap(target - rot)
                out[name + ".rot"] = rot
            accum[name] = base + rot
        for side in "lr":
            w = out.get(f"arm_{side}.ikw", 0.0)
            if w > 1e-3:
                self._reach(out, side, w)
        for k in [k for k in out if k.endswith(helpers)]:
            del out[k]
        return out

    def _reach(self, out, side, w):
        J = self.J
        up, fo = "upper_arm_" + side, "forearm_" + side
        mats = self.bone_matrices(out)
        S = _apply(mats["torso"], J["shoulder_" + side])
        T = _apply(mats["head"], self.mouth_point())
        T = T + np.array([out.get(f"arm_{side}.ikx", 0.0), out.get(f"arm_{side}.iky", 0.0)]) * self.height
        L1 = np.linalg.norm(J["elbow_" + side] - J["shoulder_" + side]) * out.get(up + ".len", 1.0)
        L2 = np.linalg.norm(self.hand_point(side) - J["elbow_" + side]) * out.get(fo + ".len", 1.0)
        v = T - S
        d0 = max(np.linalg.norm(v), 1e-6)
        u = v / d0
        d = min(max(d0, abs(L1 - L2) + 1e-3), L1 + L2 - 1e-3)
        cos_a = (L1 * L1 + d * d - L2 * L2) / (2 * L1 * d)
        sin_a = math.sqrt(max(0.0, 1 - cos_a * cos_a))
        n = np.array([-u[1], u[0]])
        cands = [S + L1 * (u * cos_a + n * sin_a), S + L1 * (u * cos_a - n * sin_a)]
        centre = _apply(mats["torso"], J["pelvis"])[0]
        sign = -1 if side == "l" else 1
        # natural arm: elbow hangs down and a little out to the side
        E = max(cands, key=lambda e: (e[1] - S[1]) + 0.5 * (e[0] - centre) * sign)
        H = S + u * d
        base = out.get("root.rot", 0.0) + out.get("torso.rot", 0.0)
        a_up = _screen_angle(E - S)
        a_fo = _screen_angle(H - E)
        rot_up = out.get(up + ".rot", 0.0)
        rot_up += w * _wrap(a_up - self.rest_angle[up] - base - rot_up)
        rot_fo = out.get(fo + ".rot", 0.0)
        rot_fo += w * _wrap(a_fo - self.rest_angle[fo] - base - rot_up - rot_fo)
        out[up + ".rot"], out[fo + ".rot"] = rot_up, rot_fo

    def _segment(self, J, solid, sw, head_len):
        """Label each pixel with its body part. Distances are measured
        *through the character's shape* (geodesic), so e.g. the side of the
        shirt isn't given to an arm hanging next to it. Done at half
        resolution for speed."""
        h, w = solid.shape
        small = cv2.resize(solid.astype(np.uint8), (max(1, w // 2), max(1, h // 2)),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
        sh, sw_ = small.shape
        ys, xs = np.mgrid[0:sh, 0:sw_]
        xs, ys = xs * 2.0 + 0.5, ys * 2.0 + 0.5
        scores = []
        for name, parent, j0, j1, th in BONES:
            thick = 0.42 * head_len if th is None else th * sw
            pen_a = 0.0 if j0 in END_JOINTS else 2.0
            pen_b = 0.0 if j1 in END_JOINTS else 2.0
            a, b = J[j0], J[j1]
            if name.startswith("thigh"):
                a = a + 0.25 * (b - a)
            if name.startswith("upper_arm"):
                # shoulder pivot sits inside the torso; start the arm region a
                # bit lower so the arm doesn't carry slivers of shirt with it
                a = a + 0.22 * (b - a)
            eucl = _seg_dist(xs, ys, a, b, pen_a, pen_b)
            seg_only = _seg_dist(xs, ys, a, b)
            seeds = (seg_only <= 2.5) & small
            limit = int(3.5 * thick / 2) + 2
            geo = _geodesic(small, seeds, limit) * 2.0
            # geodesic where reachable, plus the "past the joint" penalty
            score = np.where(np.isfinite(geo), np.maximum(geo, seg_only) + (eucl - seg_only), eucl * 3 + 1e6)
            scores.append(cv2.resize((score / thick).astype(np.float32), (w, h),
                                     interpolation=cv2.INTER_LINEAR))
        lab = np.argmin(np.stack(scores), axis=0).astype(np.uint8)
        lab = self._torso_core_fix(lab, J, solid, sw)
        hair = self._find_hair(J, solid, head_len)
        if hair is not None:
            lab[hair] = BONE_NAMES.index("head")
        # pixels right along a hand belong to that hand (fingers resting on a
        # hip or thigh would otherwise be left behind when the arm moves)
        yy, xx = np.nonzero(solid)
        for side in "lr":
            d = _seg_dist(xx, yy, J["wrist_" + side], J["hand_" + side])
            near = d < 0.11 * sw
            lab[yy[near], xx[near]] = BONE_NAMES.index("hand_" + side)
        return np.where(solid, lab, 255).astype(np.uint8)

    def _find_hair(self, J, solid, head_len):
        """Hair = pixels coloured like the top of the head and connected to
        it (above the hips). Long hair then stays with the head instead of
        being carried around by an arm."""
        h, w = solid.shape
        lab = cv2.cvtColor(self.image[..., :3], cv2.COLOR_BGR2Lab).astype(np.float32)
        top = J["head_top"]
        y0, y1 = int(max(top[1], 0)), int(min(top[1] + 0.18 * head_len, h))
        x0, x1 = int(max(top[0] - 0.3 * head_len, 0)), int(min(top[0] + 0.3 * head_len, w))
        seed = np.zeros_like(solid)
        seed[y0:y1, x0:x1] = True
        seed &= self.image[..., 3] > 200
        if seed.sum() < 20:
            return None
        colour = np.median(lab[seed], axis=0)
        wgt = np.array([0.25, 1.0, 1.0], np.float32)  # shading varies, hue doesn't
        dist = np.sqrt((((lab - colour) * wgt) ** 2).sum(axis=2))

        # adaptive threshold: stay well clear of the shirt and face colours
        def sample(c, r):
            x0, x1 = int(max(c[0] - r, 0)), int(min(c[0] + r + 1, w))
            y0, y1 = int(max(c[1] - r, 0)), int(min(c[1] + r + 1, h))
            m = self.image[y0:y1, x0:x1, 3] > 200
            return np.median(lab[y0:y1, x0:x1][m], axis=0) if m.any() else None
        chest = 0.6 * (J["shoulder_l"] + J["shoulder_r"]) / 2 + 0.4 * J["pelvis"]
        face = 0.55 * J["neck"] + 0.45 * J["head_top"]
        others = [c for c in (sample(chest, 0.08 * head_len), sample(face, 0.08 * head_len)) if c is not None]
        gap = min([np.sqrt((((c - colour) * wgt) ** 2).sum()) for c in others] + [90.0])
        similar = (dist < min(45.0, 0.5 * gap)) & solid
        hip_y = int((J["hip_l"][1] + J["hip_r"][1]) / 2)
        similar[hip_y:] = False
        similar = cv2.morphologyEx(similar.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, cc = cv2.connectedComponents(similar, connectivity=8)
        ids = np.unique(cc[seed & (similar > 0)])
        ids = ids[ids > 0]
        if not len(ids):
            return None
        hair = np.isin(cc, ids)
        hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool) & solid
        if hair.sum() > 0.4 * solid.sum():  # hair colour matches the clothes - don't trust it
            return None
        return hair

    def _torso_core_fix(self, lab, J, solid, sw):
        """Rows between shoulders and hips: the unbroken run of pixels around
        the body's centre line is torso, unless the pixel lies right on an
        arm's line (an arm crossing in front of the body)."""
        h, w = solid.shape
        core = np.zeros_like(solid)
        y_top = int(max((J["shoulder_l"][1] + J["shoulder_r"][1]) / 2, 0))
        y_bot = int(min((J["hip_l"][1] + J["hip_r"][1]) / 2, h - 1))
        p0, p1 = J["pelvis"], J["neck"]
        for y in range(y_top, y_bot + 1):
            u = (y - p0[1]) / (p1[1] - p0[1]) if p1[1] != p0[1] else 0
            cx = int(round(p0[0] + u * (p1[0] - p0[0])))
            if not (0 <= cx < w) or not solid[y, cx]:
                continue
            row = solid[y]
            left = cx
            while left > 0 and row[left - 1] and cx - left < 0.75 * sw:
                left -= 1
            right = cx
            while right < w - 1 and row[right + 1] and right - cx < 0.75 * sw:
                right += 1
            core[y, left:right + 1] = True
        ys, xs = np.nonzero(core)
        if not len(xs):
            return lab
        on_arm = np.zeros(len(xs), bool)
        for name, parent, j0, j1, th in BONES:
            if "arm" in name:
                on_arm |= _seg_dist(xs, ys, J[j0], J[j1]) < 0.55 * th * sw
        torso_i = BONE_NAMES.index("torso")
        cur = lab[ys, xs]
        arm_ids = [BONE_NAMES.index(n) for n in BONE_NAMES if "arm" in n]
        fix = np.isin(cur, arm_ids) & ~on_arm
        lab[ys[fix], xs[fix]] = torso_i
        return lab

    def _clean_labels(self, labels, alpha):
        """Smooth jagged borders and re-assign small stray islands."""
        smooth = cv2.medianBlur(labels, 5)
        labels = np.where(alpha > 0, smooth, 255).astype(np.uint8)
        labels[(alpha > 0) & (labels == 255)] = 0
        for i in range(len(BONE_NAMES)):
            m = (labels == i).astype(np.uint8)
            n, cc, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
            if n <= 2:
                continue
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            for k in range(1, n):
                if k == biggest:
                    continue
                island = cc == k
                ring = cv2.dilate(island.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~island
                neigh = labels[ring]
                neigh = neigh[(neigh != 255) & (neigh != i)]
                if len(neigh):
                    labels[island] = np.bincount(neigh).argmax()
        return labels

    def overlay_image(self):
        """BGRA picture of the parts in colour, for checking the rig."""
        out = self.image.copy()
        col = np.zeros_like(out[..., :3])
        for i, n in enumerate(BONE_NAMES):
            col[self.labels == i] = PART_COLOURS[n]
        m = self.labels != 255
        out[m, :3] = (0.55 * out[m, :3] + 0.45 * col[m]).astype(np.uint8)
        return out

    # -------------------------------------------------------------- posing
    def bone_matrices(self, pose):
        """3x3 matrices mapping rest-image coords -> posed coords (before the
        character is placed in the scene)."""
        J = self.J
        g = pose.get
        feet = J["feet"]
        root = (_T(feet) @ _R(g("root.rot", 0.0)) @ np.diag([1.0, g("root.sy", 1.0), 1.0]) @ _T(-feet))
        world = {}
        for name, parent, j0, j1, _ in BONES:
            p0, p1 = J[j0], J[j1]
            d = p1 - p0
            n = np.linalg.norm(d)
            d = d / n if n > 1e-6 else np.array([0.0, -1.0])
            k = g(name + ".len", 1.0) - 1.0
            S = np.eye(3)
            S[:2, :2] += k * np.outer(d, d)
            local = _T(p0) @ _R(g(name + ".rot", 0.0)) @ S @ _T(-p0)
            world[name] = (world[parent] if parent else root) @ local
        return world


def _wrap(a):
    return (a + 180) % 360 - 180


def _screen_angle(v):
    """0 = pointing down, 90 = right, 180 = up (counter-clockwise +)."""
    return math.degrees(math.atan2(v[0], v[1]))


def _apply(M, p):
    return (M @ np.array([p[0], p[1], 1.0]))[:2]


def _T(v):
    m = np.eye(3)
    m[0, 2], m[1, 2] = v[0], v[1]
    return m


def _R(deg):
    """Counter-clockwise on screen (y axis points down)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
