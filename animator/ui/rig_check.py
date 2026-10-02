"""'Check the dots' dialog: shows the automatic body-part split and joint
dots. The user can drag a dot if it's obviously wrong - nothing else."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from ..core.rig import Rig
from .qt_utils import bgra_to_qimage

LINES = [("head_top", "neck"), ("neck", "shoulder_l"), ("neck", "shoulder_r"),
         ("shoulder_l", "elbow_l"), ("elbow_l", "wrist_l"), ("shoulder_r", "elbow_r"),
         ("elbow_r", "wrist_r"), ("neck", "hip_l"), ("neck", "hip_r"), ("hip_l", "hip_r"),
         ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("hip_r", "knee_r"), ("knee_r", "ankle_r")]


class DotsView(QWidget):
    def __init__(self, dialog):
        super().__init__()
        self.d = dialog
        self.drag = None
        self.setMinimumSize(420, 560)

    def _xf(self):
        h, w = self.d.cutout.shape[:2]
        s = min(self.width() / w, self.height() / h)
        return s, (self.width() - w * s) / 2, (self.height() - h * s) / 2

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)
        # checkerboard so transparency is visible
        for y in range(0, self.height(), 16):
            for x in range(0, self.width(), 16):
                p.fillRect(x, y, 16, 16, QColor(205, 205, 205) if (x + y) // 16 % 2 else QColor(235, 235, 235))
        s, ox, oy = self._xf()
        h, w = self.d.cutout.shape[:2]
        img = self.d.rig.overlay_image() if self.d.show_parts else self.d.cutout
        p.drawImage(QRectF(ox, oy, w * s, h * s), bgra_to_qimage(img))
        j = self.d.joints
        p.setPen(QPen(QColor(255, 255, 255, 200), 2))
        for a, b in LINES:
            p.drawLine(QPointF(ox + j[a][0] * s, oy + j[a][1] * s), QPointF(ox + j[b][0] * s, oy + j[b][1] * s))
        for name, (x, y) in j.items():
            col = QColor(230, 40, 40) if name.endswith("_l") else QColor(40, 90, 230) if name.endswith("_r") else QColor(250, 200, 0)
            p.setPen(QPen(Qt.black, 1.5))
            p.setBrush(QBrush(col))
            p.drawEllipse(QPointF(ox + x * s, oy + y * s), 7, 7)
        p.end()

    def mousePressEvent(self, e):
        s, ox, oy = self._xf()
        best, bd = None, 15.0
        for name, (x, y) in self.d.joints.items():
            d = np.hypot(ox + x * s - e.position().x(), oy + y * s - e.position().y())
            if d < bd:
                best, bd = name, d
        self.drag = best

    def mouseMoveEvent(self, e):
        if self.drag:
            s, ox, oy = self._xf()
            self.d.joints[self.drag] = ((e.position().x() - ox) / s, (e.position().y() - oy) / s)
            self.update()

    def mouseReleaseEvent(self, e):
        if self.drag:
            self.drag = None
            self.d.rebuild()


class RigCheckDialog(QDialog):
    def __init__(self, cutout, joints, method, name, parent=None, profile=False):
        super().__init__(parent)
        self.setWindowTitle(f"Check character: {name}")
        self.cutout, self.name = cutout, name
        self.auto_joints = dict(joints)
        self.joints = dict(joints)
        self.show_parts = True
        self.profile = profile
        self.rig = Rig(cutout, self.joints, name, profile=profile)

        found = {"ai": "The body was found automatically (AI pose detection).",
                 "from_front": "Dots placed using the front view's dots (AI is unreliable on side/back views).",
                 }.get(method, "AI pose detection could not find the body, so the dots are a rough guess.")
        info = QLabel(f"<b>{found}</b><br>Coloured areas = body parts that will move. "
                      "If a dot is clearly in the wrong place, drag it onto the right joint "
                      "(red = left side of the picture, blue = right side). Otherwise just click "
                      "<b>Looks good</b>.")
        info.setWordWrap(True)
        self.view = DotsView(self)
        toggle = QPushButton("Show / hide colours")
        toggle.clicked.connect(self._toggle)
        reset = QPushButton("Reset dots")
        reset.clicked.connect(self._reset)
        ok = QPushButton("Looks good")
        ok.setDefault(True)
        ok.setStyleSheet("font-weight: bold; padding: 6px 18px;")
        ok.clicked.connect(self.accept)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons = QHBoxLayout()
        for b in (toggle, reset):
            buttons.addWidget(b)
        buttons.addStretch()
        buttons.addWidget(cancel)
        buttons.addWidget(ok)
        lay = QVBoxLayout(self)
        lay.addWidget(info)
        lay.addWidget(self.view, 1)
        lay.addLayout(buttons)
        self.resize(560, 820)

    def rebuild(self):
        self.setCursor(Qt.WaitCursor)
        self.rig = Rig(self.cutout, self.joints, self.name, profile=self.profile)
        self.unsetCursor()
        self.view.update()

    def _toggle(self):
        self.show_parts = not self.show_parts
        self.view.update()

    def _reset(self):
        self.joints = dict(self.auto_joints)
        self.rebuild()
