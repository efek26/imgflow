"""Numpy görüntülerini ölçeklenmiş şekilde gösteren önizleme widget'ı."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QScrollArea

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


def normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """Herhangi bir sayısal görüntüyü ekranda görünür 0-255 aralığına gerer.

    Eskiden `max_value <= 255` iken değer doğrudan `astype(uint8)` ile kopyalanıyordu — bu,
    `segment.connected_components`'ın "labels" çıktısı (int32, arka plan=0, bileşenler
    1,2,3,...) SEÇİLİ ADIM olarak önizlendiğinde ekranın neredeyse tamamen SİYAH görünmesine
    yol açıyordu (gerçek kullanıcı raporu: "bağlı bileşenler filtresini açınca görüntü
    kayboluyor"), çünkü piksel değeri 1-3 gibi neredeyse siyaha yakın kalıyordu. Artık min/maks
    arası her zaman 0-255'e gerilir; zaten uint8 olan görüntüler (üstteki erken dönüş) hiç
    etkilenmez."""
    if array.dtype == np.uint8:
        return array
    if array.size == 0:
        return array.astype(np.uint8)
    min_value = float(array.min())
    max_value = float(array.max())
    if max_value <= min_value:
        return np.zeros_like(array, dtype=np.uint8)
    scaled = (array.astype(np.float64) - min_value) / (max_value - min_value) * 255
    return scaled.astype(np.uint8)


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


MIN_ZOOM = 0.25
MAX_ZOOM = 8.0
_ZOOM_STEP = 1.25
"""Çarpımsal adım (ör. %25 büyüt/küçült) — toplamsal değil, böylece hem düşük hem yüksek
zoom seviyelerinde adım hep orantılı/öngörülebilir hisseder."""


class ImageView(QLabel):
    zoom_changed = Signal(float)
    image_file_dropped = Signal(str)
    """Sürüklenip bırakılan bir yerel dosyanın yolunu taşır — SADECE `setAcceptDrops(True)`
    çağrılan örneklerde tetiklenir (Qt'nin varsayılanı `acceptDrops=False`'tur), bu yüzden
    diyalog önizlemeleri gibi diğer `ImageView` kullanımları TAMAMEN etkilenmez, sürükle-bırak
    opt-in'dir (bkz. `main_window.py`'nin ana pipeline önizlemesinde nasıl açtığı)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(120, 120)
        self.setStyleSheet("background-color: #202020; color: #999;")
        self._pixmap: QPixmap | None = None
        self._zoom = 1.0
        self._scroll_host: QScrollArea | None = None
        """Set edilirse (bkz. `set_scroll_host`), zoom>1 iken bu QScrollArea'nın viewport
        boyutu "sığdır" referansı olarak kullanılır ve label bu boyuttan büyük çizilip
        kaydırma çubuklarıyla gezinilebilir hale gelir. Set edilmezse zoom metodları
        no-op'tur (eski, sabit "widget'a sığdır" davranışı korunur — ör. diyalog önizlemeleri)."""
        self.set_image(None)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        for url in event.mimeData().urls():
            if url.isLocalFile():
                self.image_file_dropped.emit(url.toLocalFile())
                event.acceptProposedAction()
                return
        event.ignore()

    def set_scroll_host(self, scroll_area: QScrollArea | None) -> None:
        self._scroll_host = scroll_area
        self._rescale()

    # -- zoom API ---------------------------------------------------------

    def zoom_in(self) -> None:
        self.set_zoom(self._zoom * _ZOOM_STEP)

    def zoom_out(self) -> None:
        self.set_zoom(self._zoom / _ZOOM_STEP)

    def zoom_reset(self) -> None:
        self.set_zoom(1.0)

    def set_zoom(self, zoom: float) -> None:
        if self._scroll_host is None:
            return
        clamped = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        if clamped == self._zoom:
            return
        self._zoom = clamped
        self._rescale()
        self.zoom_changed.emit(self._zoom)

    @property
    def zoom(self) -> float:
        return self._zoom

    def set_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._pixmap = None
            self.setPixmap(QPixmap())
            self.setText("Önizleme yok")
            return
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        self._pixmap = QPixmap.fromImage(qimage)
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._rescale()

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._scroll_host is not None and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            elif event.angleDelta().y() < 0:
                self.zoom_out()
            event.accept()
            return
        # Ctrl basılı değilse event'i YOK SAYARIZ (QLabel'in varsayılan davranışıyla aynı) —
        # bu, olayın QScrollArea'ya kabarcıklanıp normal (dikey) kaydırmayı yapmasını sağlar.
        event.ignore()

    def _rescale(self) -> None:
        if self._pixmap is None:
            return
        if self._scroll_host is not None:
            self._scroll_host.setWidgetResizable(self._zoom == 1.0)
        if self._scroll_host is None or self._zoom == 1.0:
            self.setMinimumSize(120, 120)
            self.setMaximumSize(16_777_215, 16_777_215)
            self.setPixmap(
                self._pixmap.scaled(
                    self.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            return
        # Zoom aktif: referans "sığdır" boyutu olarak KENDİ (sürekli büyüyen) boyutumuz
        # yerine sabit viewport boyutu kullanılır — aksi halde her _rescale çağrısı bir
        # önceki büyümüş boyutu baz alıp katlanarak büyür (geri besleme döngüsü).
        viewport = self._scroll_host.viewport().size()
        img_size = self._pixmap.size()
        if viewport.width() <= 0 or viewport.height() <= 0 or img_size.width() <= 0 or img_size.height() <= 0:
            return
        fit_scale = min(viewport.width() / img_size.width(), viewport.height() / img_size.height())
        scale = fit_scale * self._zoom
        target_w = max(1, int(img_size.width() * scale))
        target_h = max(1, int(img_size.height() * scale))
        self.setFixedSize(target_w, target_h)
        self.setPixmap(
            self._pixmap.scaled(
                target_w,
                target_h,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
