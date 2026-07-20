"""Numpy görüntülerini ölçeklenmiş şekilde gösteren önizleme widget'ı."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel

from imgflow.core.types import PortType


def numpy_to_qimage(image: np.ndarray) -> QImage:
    array = np.ascontiguousarray(image)
    if array.ndim == 2:
        h, w = array.shape
        return QImage(array.data, w, h, array.strides[0], QImage.Format.Format_Grayscale8).copy()
    if array.ndim == 3 and array.shape[2] == 3:
        h, w, _ = array.shape
        rgb = np.ascontiguousarray(array[:, :, ::-1])  # BGR -> RGB
        return QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888).copy()
    raise ValueError(f"Desteklenmeyen görüntü şekli: {array.shape}")


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    if array.dtype == np.uint8:
        return array
    max_value = float(array.max()) if array.size else 0.0
    if max_value <= 0:
        return np.zeros_like(array, dtype=np.uint8)
    if max_value <= 255:
        return array.astype(np.uint8)
    return (array.astype(np.float64) / max_value * 255).astype(np.uint8)


def extract_preview_image(op_cls: type, outputs: dict[str, Any] | None) -> np.ndarray | None:
    """op_cls.outputs içindeki ilk IMAGE tipli portun değerini döner (yoksa None)."""
    if not outputs:
        return None
    for port in op_cls.outputs:
        if port.type is PortType.IMAGE:
            value = outputs.get(port.name)
            if isinstance(value, np.ndarray):
                return value
    return None


class ImageView(QLabel):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 120)
        self.setStyleSheet("background-color: #202020; color: #999;")
        self._pixmap: QPixmap | None = None
        self.set_image(None)

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText("Önizleme yok")
            return
        qimage = numpy_to_qimage(_normalize_to_uint8(image))
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        self.setPixmap(
            self._pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
