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
    ("forearm_l", "upper_arm_l", "elbow_l", "hand_l", 0.20),
    ("upper_arm_r", "torso", "shoulder_r", "elbow_r", 0.20),
    ("forearm_r", "upper_arm_r", "elbow_r", "hand_r", 0.20),
    ("thigh_l", None, "hip_l", "knee_l", 0.26),
    ("shin_l", "thigh_l", "knee_l", "foot_l", 0.21),
    ("thigh_r", None, "hip_r", "knee_r", 0.26),
    ("shin_r", "thigh_r", "knee_r", "foot_r", 0.21),
]
BONE_NAMES = [b[0] for b in BONES]
DRAW_ORDER = ["thigh_l", "thigh_r", "shin_l", "shin_r", "torso", "head",
              "upper_arm_l", "upper_arm_r", "forearm_l", "forearm_r"]
PART_COLOURS = {  # for the "check the dots" overlay (BGR)
    "torso": (60, 180, 75), "head": (25, 225, 255), "upper_arm_l": (200, 130, 0),
    "forearm_l": (240, 50, 230), "upper_arm_r": (48, 130, 245), "forearm_r": (180, 30, 145),
    "thigh_l": (75, 25, 230), "shin_l": (128, 128, 0), "thigh_r": (0, 128, 128), "shin_r": (195, 255, 170),
}


def derived_joints(joints, alpha):
    """Adds pelvis, hands, feet and ground point to the user-facing joints."""
    j = {k: np.array(v, float) for k, v in joints.items()}
    j["pelvis"] = (j["hip_l"] + j["hip_r"]) / 2
    for s in "lr":
        j["hand_" + s] = j["wrist_" + s] + 0.4 * (j["wrist_" + s] - j["elbow_" + s])
    ys, xs = np.nonzero(alpha > 128)
    ground = float(ys.max())
    for s in "lr":
        ankle = j["ankle_" + s]
        j["foot_" + s] = np.array([ankle[0], max(ground, ankle[1] + 1)])
    j["feet"] = np.array([j["pelvis"][0], ground])
    return j


END_JOINTS = {"pelvis", "head_top", "hand_l", "hand_r", "foot_l", "foot_r"}


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

    def __init__(self, rgba, joints, name="Character"):
        self.name = name
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
            cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([tx, ty], 1).astype(np.int32)), 1)
        hull = hull.astype(bool) & solid
        arms = np.isin(labels, [idx[n] for n in ("upper_arm_l", "forearm_l", "upper_arm_r", "forearm_r")])
        behind = hull & arms
        torso_img = self.image.copy()
        if behind.any():
            torso_img[..., :3] = cv2.inpaint(self.image[..., :3], behind.astype(np.uint8) * 255,
                                             7, cv2.INPAINT_TELEA)
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
            levels = [(crop, float(x0 - 1), float(y0 - 1), 1.0)]
            f = 1.0
            while min(crop.shape[:2]) > 8 and f < 8:
                crop = cv2.resize(crop, (max(1, crop.shape[1] // 2), max(1, crop.shape[0] // 2)),
                                  interpolation=cv2.INTER_AREA)
                f *= 2
                levels.append((crop, float(x0 - 1), float(y0 - 1), f))
            self.parts[name] = levels

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
        return np.where(solid, lab, 255).astype(np.uint8)

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


def _T(v):
    m = np.eye(3)
    m[0, 2], m[1, 2] = v[0], v[1]
    return m


def _R(deg):
    """Counter-clockwise on screen (y axis points down)."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])
