"""The scene preview: shows the rendered frame and lets you select, drag,
and resize characters, or click a walk destination."""
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from ..core.scene import to_bgr_over
from .qt_utils import bgr_to_qimage

PREVIEW_W, PREVIEW_H = 960, 540


class PreviewWidget(QWidget):
    selectionChanged = Signal(object)   # Character or None
    sceneEdited = Signal()
    targetPicked = Signal(float, float)  # normalised scene coords

    def __init__(self, scene, parent=None):
        super().__init__(parent)
        self.scene = scene
        self.time = 0.0
        self.selected = None
        self.picking_target = False
        self._image = None
        self._boxes = {}
        self._drag = None
        self.setMinimumSize(480, 270)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------ drawing
    def refresh(self):
        frame, boxes = self.scene.render(self.time, PREVIEW_W, PREVIEW_H)
        self._image = bgr_to_qimage(to_bgr_over(frame))
        self._boxes = boxes
        self.update()

    def _view_rect(self):
        w, h = self.width(), self.height()
        s = min(w / PREVIEW_W, h / PREVIEW_H)
        vw, vh = PREVIEW_W * s, PREVIEW_H * s
        return QRectF((w - vw) / 2, (h - vh) / 2, vw, vh)

    def paintEvent(self, _):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(40, 40, 40))
        r = self._view_rect()
        if self._image is not None:
            p.setRenderHint(QPainter.SmoothPixmapTransform)
            p.drawImage(r, self._image)
        sx = r.width() / PREVIEW_W
        if self.selected is not None and id(self.selected) in self._boxes and self._boxes[id(self.selected)]:
            x0, y0, x1, y1 = self._boxes[id(self.selected)]
            p.setPen(QPen(QColor(255, 210, 0), 2, Qt.DashLine))
            p.drawRect(QRectF(r.x() + x0 * sx, r.y() + y0 * sx, (x1 - x0) * sx, (y1 - y0) * sx))
        if not self.scene.characters:
            p.setPen(QColor(255, 255, 255))
            p.setFont(QFont("Segoe UI", 16))
            p.drawText(r, Qt.AlignCenter, "Click  “Add character”  to begin")
        if self.picking_target:
            p.setPen(QColor(255, 255, 0))
            p.setFont(QFont("Segoe UI", 13, QFont.Bold))
            p.drawText(r.adjusted(0, 10, 0, 0), Qt.AlignHCenter | Qt.AlignTop,
                       "Click where the character should walk to  (Esc = cancel)")
        p.end()

    # ------------------------------------------------------------ mouse
    def _to_scene(self, pos):
        r = self._view_rect()
        return ((pos.x() - r.x()) / r.width(), (pos.y() - r.y()) / r.height())

    def _hit(self, pos):
        r = self._view_rect()
        sx = r.width() / PREVIEW_W
        px, py = (pos.x() - r.x()) / sx, (pos.y() - r.y()) / sx
        for ch in reversed(self.scene.characters):  # top-most first
            b = self._boxes.get(id(ch))
            if b and b[0] <= px <= b[2] and b[1] <= py <= b[3]:
                return ch
        return None

    def mousePressEvent(self, e):
        if e.button() != Qt.LeftButton:
            return
        if self.picking_target:
            x, y = self._to_scene(e.position())
            self.picking_target = False
            self.targetPicked.emit(min(max(x, 0.0), 1.0), min(max(y, 0.0), 1.0))
            self.update()
            return
        ch = self._hit(e.position())
        if ch is not self.selected:
            self.selected = ch
            self.selectionChanged.emit(ch)
        if ch is not None:
            self._drag = (self._to_scene(e.position()), tuple(ch.pos),
                          [a.target for a in ch.actions])
        self.update()

    def mouseMoveEvent(self, e):
        if self._drag and self.selected is not None:
            (x0, y0), (px, py), targets = self._drag
            x, y = self._to_scene(e.position())
            dx, dy = x - x0, y - y0
            self.selected.pos = (px + dx, py + dy)
            # move the whole path with the character
            for a, t in zip(self.selected.actions, targets):
                if t is not None:
                    a.target = (t[0] + dx, t[1] + dy)
            self.refresh()
            self.sceneEdited.emit()

    def mouseReleaseEvent(self, e):
        self._drag = None

    def wheelEvent(self, e):
        if self.selected is None:
            return
        steps = e.angleDelta().y() / 120
        self.selected.height = min(max(self.selected.height * (1.06 ** steps), 0.05), 2.0)
        self.refresh()
        self.sceneEdited.emit()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self.picking_target:
            self.picking_target = False
            self.update()
