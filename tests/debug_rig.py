"""Build a rig from an image (or the synthetic character) and save an overlay."""
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from animator.core.imageio_utils import read_image  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.rig import Rig  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from make_test_character import make_character  # noqa: E402


def overlay(rig, joints):
    ov = rig.overlay_image()
    for x, y in joints.values():
        cv2.circle(ov, (int(x), int(y)), 6, (0, 0, 255, 255), -1)
    a = ov[..., 3:4] / 255
    return (ov[..., :3] * a + 200 * (1 - a)).astype(np.uint8)


if __name__ == "__main__":
    img = read_image(sys.argv[1]) if len(sys.argv) > 1 else make_character()
    cut = prepare_cutout(img)
    j, method = detect_joints(cut)
    print("joints:", method)
    rig = Rig(cut, j)
    out = Path("test_output")
    out.mkdir(exist_ok=True)
    cv2.imwrite(str(out / "rig_overlay.png"), overlay(rig, j))
