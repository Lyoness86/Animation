"""Off-screen UI smoke test: builds the window with two characters and saves
screenshots (QT_QPA_PLATFORM=offscreen python tests/ui_smoke.py)."""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
from PySide6.QtCore import QPointF, Qt  # noqa: E402
from PySide6.QtGui import QMouseEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from animator.core.character import Action, Character  # noqa: E402
from animator.core.keying import prepare_cutout  # noqa: E402
from animator.core.skeleton import detect_joints  # noqa: E402
from animator.ui.main_window import MainWindow  # noqa: E402
from animator.ui.rig_check import RigCheckDialog  # noqa: E402
from make_test_character import make_character  # noqa: E402

app = QApplication(sys.argv)
win = MainWindow()
win.show()
out = Path("test_output")
out.mkdir(exist_ok=True)

bg = np.zeros((1080, 1920, 3), np.uint8)
bg[:700] = (235, 200, 140)
bg[700:] = (70, 160, 90)
win.scene.set_background(bg)

for i, shirt in enumerate([(40, 60, 200), (60, 160, 40)]):
    cut = prepare_cutout(make_character(shirt=shirt, seed=i))
    j, method = detect_joints(cut)
    dlg = RigCheckDialog(cut, j, method, f"Character {i + 1}", win)
    dlg.show()
    app.processEvents()
    if i == 0:
        dlg.grab().save(str(out / "ui_rigcheck.png"))
    dlg.accept()
    ch = Character(dlg.rig, pos=(0.25 + 0.45 * i, 0.95), height=0.6)
    win.scene.characters.append(ch)
    win._select(ch)

a, b = win.scene.characters
win._select(a)
# simulate "Walk to..." + click on the preview
win.start_walk_pick()
r = win.preview._view_rect()
pt = QPointF(r.x() + r.width() * 0.55, r.y() + r.height() * 0.95)
win.preview.mousePressEvent(QMouseEvent(QMouseEvent.MouseButtonPress, pt, pt, Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
win.add_action(Action("wave"))
win._select(b)
win.add_action(Action("jump"))
win.add_action(Action("idle", duration=1.0))
win.add_action(Action("wave"))
assert [x.kind for x in a.actions] == ["walk", "stop", "wave"], a.actions
win._select(a)
win.set_time(1.0)
app.processEvents()
win.grab().save(str(out / "ui_main.png"))
print("duration", win.scene.duration(), "OK")
