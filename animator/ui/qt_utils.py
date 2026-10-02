import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap


def bgr_to_qimage(bgr):
    bgr = np.ascontiguousarray(bgr)
    h, w = bgr.shape[:2]
    return QImage(bgr.data, w, h, 3 * w, QImage.Format_BGR888).copy()


def bgra_to_qimage(bgra):
    """Straight-alpha BGRA uint8 -> QImage."""
    bgra = np.ascontiguousarray(bgra)
    h, w = bgra.shape[:2]
    return QImage(bgra.data, w, h, 4 * w, QImage.Format_ARGB32).copy()


def thumbnail(bgra, size=48):
    return QPixmap.fromImage(bgra_to_qimage(bgra)).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
