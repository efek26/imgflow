"""ImageView'i genişletip fare ile ROI çizme/taşıma/yeniden boyutlandırma ekler.

Sadece editing_enabled=True iken (roi.region adımı seçiliyken) aktiftir. Görüntü, en-boy
oranı korunarak ortalanmış şekilde ölçekli gösterildiği için, fare tıklamalarını gerçek
görüntü piksel koordinatına çevirmek üzere kendi dönüşümünü (_display_rect) yapar.

Üç ROI şekli desteklenir (set_shape ile seçilir): "RECT" (köşe tutamaçlarıyla dikdörtgen,
varsayılan/geriye dönük uyumlu davranış), "CIRCLE" (merkezden sürükleyerek çizilen, tek bir
kenar tutamacıyla yarıçapı değiştirilebilen daire) ve "POLYGON" (serbest kontur/poligon —
`ShapeMatchingDialog`'un "Serbest Kontur" modu için: kapanmamışken sol tık YENİ nokta ekler,
sağ tık son noktayı geri alır; `close_polygon()`/`clear_polygon()` ile kapatılır/sıfırlanır;
kapandıktan sonra bir köşeye yakın sürükleme onu taşır).

Ayrıca `set_multi_mode(True)` ile AYRI bir "çoklu ROI" modu vardır (`analysis.color_props`/
`analysis.texture_props`'un "Elle ROI Çiz" parametresi için, bkz. `ui/main_window.py`):
tek bir dikdörtgen yerine kullanıcının istediği kadar dikdörtgen çizip taşıyabildiği/yeniden
boyutlandırabildiği bir liste yönetilir -- boş alana sürüklemek YENİ bir ROI ekler, mevcut
bir ROI'nin içine sürüklemek onu taşır, köşe tutamacı onu yeniden boyutlandırır, ÜZERİNE SAĞ
TIKLAMAK siler. Bu iki mod (tekli `set_shape`/çoklu `set_multi_mode`) aynı anda AKTİF
OLMAZ -- `main_window.py` hangi operatör seçiliyse ona göre birini açar.

`set_contour_preview(excluded_mask)` (RECT/POLYGON/CIRCLE üç modunda da çizilir) --
`ShapeMatchingDialog`'un
"Konturu Otomatik Algıla" checkbox'ı işaretliyken ROI içinde HANGİ piksellerin modelden
ELENECEĞİNİ (gerçek nesnenin dış hattı DIŞINDaki arka plan) yarı saydam kırmızı bir katmanla
gösterir -- gerçek kullanıcı raporu: "elediği arka planı göremiyorum, onu da görmek
istiyorum". `excluded_mask`, GÖRÜNTÜ ile aynı (yükseklik, genişlik) boyutunda bool dizi (True
= elenecek/kırmızı gösterilecek piksel); `None` önizlemeyi temizler.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QWidget

from imgflow.ui.widgets import _display_transform as dt
from imgflow.ui.widgets.image_view import ImageView

_HANDLE_HIT_RADIUS = 8
_ROI_COLOR = QColor("#33cc33")
_EXCLUDED_OVERLAY_RGBA = (255, 40, 40, 130)
"""Kontur önizlemesinde "elenecek" (arka plan) piksellerin rengi -- yarı saydam kırmızı,
kalan nesne dış hattıyla (yeşil ROI çerçevesi) net bir zıtlık için."""
_MIN_MULTI_ROI_SIZE = 4
"""Yeni bir çoklu-ROI dikdörtgeni bu boyuttan (px, görüntü koordinatında) KÜÇÜKSE eklenmez --
aksi halde sürüklemeden basit bir tıklama bile yanlışlıkla 1x1'lik bir ROI oluşturur."""


class RoiCanvas(ImageView):
    roi_changed = Signal(int, int, int, int)  # x, y, w, h - GÖRÜNTÜ koordinatında
    roi_circle_changed = Signal(int, int, int)  # cx, cy, r - GÖRÜNTÜ koordinatında
    rois_changed = Signal(object)  # list[tuple[int,int,int,int]] - GÖRÜNTÜ koordinatında (çoklu mod)
    polygon_changed = Signal(object)  # list[tuple[int,int]] - GÖRÜNTÜ koordinatında (POLYGON modu)
    hover_measurement_changed = Signal(object)  # dict | None - fare bir ölçüm kutusunun üzerindeyken

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self._image_size = (0, 0)  # (genişlik, yükseklik) - orijinal görüntü boyutu
        self._shape = "RECT"  # "RECT" | "CIRCLE" | "POLYGON"
        self._roi_image: tuple[int, int, int, int] | None = None
        self._roi_circle: tuple[int, int, int] | None = None  # cx, cy, r
        self._editing_enabled = False
        self._drag_mode: str | None = None  # None | "new" | "move" | "resize:tl" | "resize:r" | ...
        self._drag_anchor = QPoint()
        self._drag_start_rect = QRect()
        self._drag_start_center = QPoint()
        self._drag_start_radius = 0
        self._polygon_points: list[tuple[int, int]] = []
        """POLYGON modunda GÖRÜNTÜ koordinatındaki köşeler, ekleniş sırasıyla."""
        self._polygon_closed = False
        self._polygon_drag_index: int | None = None
        self._contour_preview: np.ndarray | None = None
        """`set_contour_preview` ile beslenir -- GÖRÜNTÜ boyutunda bool dizi, True olan
        pikseller "Konturu Otomatik Algıla" tarafından elenecek/kırmızı gösterilecek
        (üç çizim modunda da)."""
        self._multi_mode = False
        self._rois: list[tuple[int, int, int, int]] = []
        """Çoklu-ROI modunda GÖRÜNTÜ koordinatındaki dikdörtgenler -- listedeki sıra ROI
        numarasını (1-tabanlı) belirler, `main_window.py` bunu `manual_rois` parametresine
        JSON olarak senkronize eder."""
        self._multi_drag_mode: str | None = None  # None | "new" | "move" | "resize:tl" | ...
        self._multi_drag_index = -1  # düzenlenen ROI'nin self._rois içindeki index'i ("new" iken -1)
        self._multi_drag_start_rect = QRect()
        self._multi_live_rect: QRect | None = None  # sürükleme sırasında WIDGET koordinatında canlı önizleme
        self._measurements: list[dict] = []
        """`set_measurements` ile beslenir -- `bbox_x/y/w/h` alanları GÖSTERİLEN görüntünün
        koordinat sisteminde olmalı (ör. küçültülmüş önizlemede orana göre zaten ölçeklenmiş,
        bkz. `ui/main_window.py` `_scale_measurements_for_display`)."""
        self._hover_measurement: dict | None = None
        self._last_mouse_pos: QPoint | None = None
        """Son bilinen fare pozisyonu (widget koordinatında) -- her canlı kamera tick'inde
        (~10Hz) `set_measurements` çağrıldığında, fare HİÇ HAREKET ETMESE bile üzerinde
        durduğu nesnenin bilgisini güncel tutabilmek için (gerçek OS imleç konumunu sorgulamak
        yerine bunu saklamak, headless/offscreen test ortamında da güvenilir çalışır)."""

    def set_measurements(self, measurements: list[dict] | None) -> None:
        self._measurements = measurements or []
        if self._last_mouse_pos is not None:
            self._update_hover(self._last_mouse_pos)

    # -- genel API ------------------------------------------------------

    def set_image(self, image) -> None:  # noqa: ANN001 - ImageView ile aynı imza
        super().set_image(image)
        self._image_size = (image.shape[1], image.shape[0]) if image is not None else (0, 0)
        self.update()

    def set_editing_enabled(self, enabled: bool) -> None:
        self._editing_enabled = enabled
        self.update()

    def set_shape(self, shape: str) -> None:
        self._shape = shape
        self.update()

    def set_roi(self, x: int, y: int, w: int, h: int) -> None:
        self._roi_image = (x, y, w, h)
        self.update()

    def set_roi_circle(self, cx: int, cy: int, r: int) -> None:
        self._roi_circle = (cx, cy, r)
        self.update()

    def set_multi_mode(self, enabled: bool) -> None:
        self._multi_mode = enabled
        self._multi_drag_mode = None
        self._multi_live_rect = None
        self.update()

    def set_rois(self, rois: list[tuple[int, int, int, int]]) -> None:
        self._rois = list(rois)
        self.update()

    def close_polygon(self) -> None:
        """En az 3 nokta varsa poligonu kapatır (`polygon_changed` yayınlar); az noktayla
        çağrılırsa sessizce hiçbir şey yapmaz (dialog, buton `enabled` durumunu nokta
        sayısına göre zaten ayarlar)."""
        if len(self._polygon_points) >= 3:
            self._polygon_closed = True
            self.polygon_changed.emit(list(self._polygon_points))
            self.update()

    def clear_polygon(self) -> None:
        self._polygon_points = []
        self._polygon_closed = False
        self._polygon_drag_index = None
        self.update()

    def polygon_points(self) -> list[tuple[int, int]]:
        return list(self._polygon_points)

    def is_polygon_closed(self) -> bool:
        return self._polygon_closed

    def set_contour_preview(self, excluded_mask: np.ndarray | None) -> None:
        self._contour_preview = excluded_mask
        self.update()

    # -- koordinat dönüşümü (görüntü <-> widget) -----------------------------

    def _widget_size(self) -> tuple[int, int]:
        return (self.width(), self.height())

    def _display_rect(self) -> QRect:
        """Ölçeklenmiş görüntünün widget içindeki (ortalanmış) dikdörtgeni."""
        return dt.display_rect(self._image_size, self._widget_size())

    def _roi_widget_rect(self) -> QRect | None:
        if self._roi_image is None:
            return None
        return dt.image_rect_to_widget(self._roi_image, self._image_size, self._widget_size())

    def _widget_rect_to_image(self, rect: QRect) -> tuple[int, int, int, int]:
        return dt.widget_rect_to_image(rect, self._image_size, self._widget_size())

    def _circle_widget(self) -> tuple[QPoint, int] | None:
        if self._roi_circle is None:
            return None
        cx, cy, r = self._roi_circle
        return dt.image_circle_to_widget(cx, cy, r, self._image_size, self._widget_size())

    def _widget_circle_to_image(self, center: QPoint, radius: int) -> tuple[int, int, int]:
        return dt.widget_circle_to_image(center, radius, self._image_size, self._widget_size())

    @staticmethod
    def _dist(a: QPoint, b: QPoint) -> float:
        dx, dy = a.x() - b.x(), a.y() - b.y()
        return (dx * dx + dy * dy) ** 0.5

    def _handle_at_circle(self, pos: QPoint) -> str | None:
        widget_circle = self._circle_widget()
        if widget_circle is None:
            return None
        center, radius = widget_circle
        edge = QPoint(center.x() + radius, center.y())
        if (edge - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
            return "r"
        return None

    @staticmethod
    def _handle_points(rect: QRect) -> dict[str, QPoint]:
        return {
            "tl": rect.topLeft(),
            "tr": rect.topRight(),
            "bl": rect.bottomLeft(),
            "br": rect.bottomRight(),
        }

    def _handle_at(self, pos: QPoint) -> str | None:
        rect = self._roi_widget_rect()
        if rect is None:
            return None
        for name, hp in self._handle_points(rect).items():
            if (hp - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return name
        return None

    @staticmethod
    def _resize_rect(start: QRect, handle: str, delta: QPoint) -> QRect:
        r = QRect(start)
        if "t" in handle:
            r.setTop(start.top() + delta.y())
        if "b" in handle:
            r.setBottom(start.bottom() + delta.y())
        if "l" in handle:
            r.setLeft(start.left() + delta.x())
        if "r" in handle:
            r.setRight(start.right() + delta.x())
        return r.normalized()

    # -- fare olayları ----------------------------------------------------

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if not self._editing_enabled or self._image_size == (0, 0):
            super().mousePressEvent(event)
            return
        pos = event.position().toPoint()

        if self._multi_mode:
            self._mouse_press_multi(pos, event.button())
            self._drag_anchor = pos
            event.accept()
            return

        if self._shape == "POLYGON":
            self._mouse_press_polygon(pos, event.button())
            event.accept()
            return

        if self._shape == "CIRCLE":
            self._mouse_press_circle(pos)
        else:
            self._mouse_press_rect(pos)
        self._drag_anchor = pos
        event.accept()

    def _multi_rect_at(self, pos: QPoint) -> int | None:
        """En üstteki (son çizilen) ROI önce sınanır -- üst üste binen ROI'lerde en son
        eklenen/en son taşınan seçilsin diye."""
        for i in reversed(range(len(self._rois))):
            rect = dt.image_rect_to_widget(self._rois[i], self._image_size, self._widget_size())
            if rect.contains(pos):
                return i
        return None

    def _handle_at_rect(self, pos: QPoint, rect: QRect) -> str | None:
        for name, hp in self._handle_points(rect).items():
            if (hp - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return name
        return None

    def _mouse_press_multi(self, pos: QPoint, button: Qt.MouseButton) -> None:
        if button == Qt.MouseButton.RightButton:
            index = self._multi_rect_at(pos)
            if index is not None:
                del self._rois[index]
                self.rois_changed.emit(list(self._rois))
                self.update()
            return

        for i in reversed(range(len(self._rois))):
            rect = dt.image_rect_to_widget(self._rois[i], self._image_size, self._widget_size())
            handle = self._handle_at_rect(pos, rect)
            if handle is not None:
                self._multi_drag_mode = f"resize:{handle}"
                self._multi_drag_index = i
                self._multi_drag_start_rect = rect
                return

        index = self._multi_rect_at(pos)
        if index is not None:
            self._multi_drag_mode = "move"
            self._multi_drag_index = index
            self._multi_drag_start_rect = dt.image_rect_to_widget(
                self._rois[index], self._image_size, self._widget_size()
            )
            return

        self._multi_drag_mode = "new"
        self._multi_drag_index = -1
        self._multi_drag_start_rect = QRect(pos, pos)

    def _mouse_press_rect(self, pos: QPoint) -> None:
        handle = self._handle_at(pos)
        current = self._roi_widget_rect()
        if handle is not None:
            self._drag_mode = f"resize:{handle}"
            self._drag_start_rect = current
        elif current is not None and current.contains(pos):
            self._drag_mode = "move"
            self._drag_start_rect = current
        else:
            self._drag_mode = "new"
            self._drag_start_rect = QRect(pos, pos)

    def _vertex_at(self, pos: QPoint) -> int | None:
        for i, img_pt in enumerate(self._polygon_points):
            widget_pt = dt.image_point_to_widget(img_pt, self._image_size, self._widget_size())
            if (widget_pt - pos).manhattanLength() <= _HANDLE_HIT_RADIUS:
                return i
        return None

    def _mouse_press_polygon(self, pos: QPoint, button: Qt.MouseButton) -> None:
        if button == Qt.MouseButton.RightButton:
            if not self._polygon_closed and self._polygon_points:
                self._polygon_points.pop()
                self.polygon_changed.emit(list(self._polygon_points))
                self.update()
            return
        if self._polygon_closed:
            self._polygon_drag_index = self._vertex_at(pos)
            return
        self._polygon_points.append(dt.widget_point_to_image(pos, self._image_size, self._widget_size()))
        self.polygon_changed.emit(list(self._polygon_points))
        self.update()

    def _mouse_press_circle(self, pos: QPoint) -> None:
        handle = self._handle_at_circle(pos)
        current = self._circle_widget()
        if handle is not None:
            self._drag_mode = "resize:r"
            self._drag_start_center, self._drag_start_radius = current
        elif current is not None and self._dist(pos, current[0]) <= current[1]:
            self._drag_mode = "move"
            self._drag_start_center, self._drag_start_radius = current
        else:
            self._drag_mode = "new"
            self._drag_start_center, self._drag_start_radius = pos, 0

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        pos = event.position().toPoint()
        self._last_mouse_pos = pos
        self._update_hover(pos)

        if not self._editing_enabled:
            super().mouseMoveEvent(event)
            return

        if self._multi_mode:
            if self._multi_drag_mode is None:
                super().mouseMoveEvent(event)
                return
            self._mouse_move_multi(pos)
            self.update()
            event.accept()
            return

        if self._shape == "POLYGON":
            self._mouse_move_polygon(pos)
            self.update()
            event.accept()
            return

        if self._drag_mode is None:
            super().mouseMoveEvent(event)
            return

        if self._shape == "CIRCLE":
            self._mouse_move_circle(pos)
        else:
            self._mouse_move_rect(pos)
        self.update()
        event.accept()

    def _mouse_move_multi(self, pos: QPoint) -> None:
        delta = pos - self._drag_anchor
        if self._multi_drag_mode == "new":
            rect = QRect(self._drag_anchor, pos).normalized()
        elif self._multi_drag_mode == "move":
            rect = self._multi_drag_start_rect.translated(delta)
        else:
            handle = self._multi_drag_mode.split(":", 1)[1]
            rect = self._resize_rect(self._multi_drag_start_rect, handle, delta)
        self._multi_live_rect = rect

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt override
        self._last_mouse_pos = None
        self._set_hover(None)
        super().leaveEvent(event)

    # -- üzerine gelme (hover) bilgisi ------------------------------------

    def _update_hover(self, pos: QPoint) -> None:
        if not self._measurements or self._image_size == (0, 0):
            self._set_hover(None)
            return
        widget_size = self._widget_size()
        for measurement in self._measurements:
            try:
                rect = dt.image_rect_to_widget(
                    (
                        measurement["bbox_x"],
                        measurement["bbox_y"],
                        measurement["bbox_w"],
                        measurement["bbox_h"],
                    ),
                    self._image_size,
                    widget_size,
                )
            except KeyError:
                continue
            if rect.contains(pos):
                self._set_hover(measurement)
                return
        self._set_hover(None)

    def _set_hover(self, measurement: dict | None) -> None:
        if measurement is self._hover_measurement:
            return
        self._hover_measurement = measurement
        self.hover_measurement_changed.emit(measurement)

    def _mouse_move_polygon(self, pos: QPoint) -> None:
        if self._polygon_drag_index is not None:
            self._polygon_points[self._polygon_drag_index] = dt.widget_point_to_image(
                pos, self._image_size, self._widget_size()
            )
        # Kapanmamışken hiçbir şey taşınmasa bile update() zaten çağrılıyor (çağıran
        # `mouseMoveEvent`) -- son nokta ile imleç arasındaki "kauçuk bant" çizgisi böyle
        # canlı kalır (bkz. `_paint_polygon`).

    def _mouse_move_rect(self, pos: QPoint) -> None:
        delta = pos - self._drag_anchor
        if self._drag_mode == "new":
            rect = QRect(self._drag_anchor, pos).normalized()
        elif self._drag_mode == "move":
            rect = self._drag_start_rect.translated(delta)
        else:
            handle = self._drag_mode.split(":", 1)[1]
            rect = self._resize_rect(self._drag_start_rect, handle, delta)
        self._roi_image = self._widget_rect_to_image(rect)

    def _mouse_move_circle(self, pos: QPoint) -> None:
        if self._drag_mode == "new":
            center, radius = self._drag_anchor, int(self._dist(pos, self._drag_anchor))
        elif self._drag_mode == "move":
            delta = pos - self._drag_anchor
            center, radius = self._drag_start_center + delta, self._drag_start_radius
        else:  # resize:r
            center, radius = self._drag_start_center, int(self._dist(pos, self._drag_start_center))
        self._roi_circle = self._widget_circle_to_image(center, max(radius, 1))

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt override
        if self._multi_mode:
            if self._multi_drag_mode is not None:
                self._mouse_release_multi()
                event.accept()
                return
            super().mouseReleaseEvent(event)
            return

        if self._shape == "POLYGON":
            if self._polygon_drag_index is not None:
                self._polygon_drag_index = None
                self.polygon_changed.emit(list(self._polygon_points))
                event.accept()
                return
            super().mouseReleaseEvent(event)
            return

        if self._drag_mode is not None:
            self._drag_mode = None
            if self._shape == "CIRCLE":
                if self._roi_circle is not None:
                    self.roi_circle_changed.emit(*self._roi_circle)
            elif self._roi_image is not None:
                self.roi_changed.emit(*self._roi_image)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _mouse_release_multi(self) -> None:
        mode = self._multi_drag_mode
        widget_rect = self._multi_live_rect
        index = self._multi_drag_index
        self._multi_drag_mode = None
        self._multi_live_rect = None
        if widget_rect is None:
            self.update()
            return

        img_rect = self._widget_rect_to_image(widget_rect)
        if mode == "new":
            if img_rect[2] >= _MIN_MULTI_ROI_SIZE and img_rect[3] >= _MIN_MULTI_ROI_SIZE:
                self._rois.append(img_rect)
        elif 0 <= index < len(self._rois):
            self._rois[index] = img_rect
        self.rois_changed.emit(list(self._rois))
        self.update()

    # -- çizim -------------------------------------------------------------

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt override
        super().paintEvent(event)
        if not self._editing_enabled:
            return
        painter = QPainter(self)
        painter.setPen(QPen(_ROI_COLOR, 2))

        if self._multi_mode:
            self._paint_multi(painter)
            painter.end()
            return

        # Elenen-alan önizlemesi ÜÇ çizim modunda da (RECT/POLYGON/CIRCLE) ve şeklin ALTINDA
        # çizilir -- gerçek kullanıcı raporu: "poligon ve çember ROI'nin içinde kontür
        # bulamıyor" (aslında kontur BULUNUYORDU, sadece yalnızca RECT modunda GÖSTERİLİYORDU,
        # bu yüzden diğer modlarda hiç geri bildirim olmadığından "bulamıyor" gibi görünüyordu).
        if self._contour_preview is not None:
            self._paint_contour_preview(painter)

        if self._shape == "POLYGON":
            self._paint_polygon(painter)
            painter.end()
            return

        if self._shape == "CIRCLE":
            widget_circle = self._circle_widget()
            if widget_circle is not None:
                center, radius = widget_circle
                painter.drawEllipse(center, radius, radius)
                painter.fillRect(center.x() - 3, center.y() - 3, 6, 6, _ROI_COLOR)
                edge = QPoint(center.x() + radius, center.y())
                painter.fillRect(edge.x() - 4, edge.y() - 4, 8, 8, _ROI_COLOR)
        else:
            rect = self._roi_widget_rect()
            if rect is not None:
                painter.drawRect(rect)
                for hp in self._handle_points(rect).values():
                    painter.fillRect(hp.x() - 4, hp.y() - 4, 8, 8, _ROI_COLOR)
        painter.end()

    def _paint_contour_preview(self, painter: QPainter) -> None:
        if self._contour_preview is None or self._image_size == (0, 0):
            return
        height, width = self._contour_preview.shape
        rgba = np.zeros((height, width, 4), dtype=np.uint8)
        rgba[self._contour_preview] = _EXCLUDED_OVERLAY_RGBA
        qimage = QImage(rgba.data, width, height, width * 4, QImage.Format.Format_RGBA8888).copy()
        painter.drawImage(self._display_rect(), qimage)

    def _paint_polygon(self, painter: QPainter) -> None:
        widget_points = [
            dt.image_point_to_widget(p, self._image_size, self._widget_size()) for p in self._polygon_points
        ]
        for i in range(len(widget_points) - 1):
            painter.drawLine(widget_points[i], widget_points[i + 1])
        if self._polygon_closed and len(widget_points) >= 3:
            painter.drawLine(widget_points[-1], widget_points[0])
        elif widget_points and self._last_mouse_pos is not None:
            # Kapanmamışken son köşe ile imleç arasında canlı bir "kauçuk bant" çizgisi --
            # kullanıcının nereye tıklarsa poligonu kapatacağını görmesi için.
            painter.drawLine(widget_points[-1], self._last_mouse_pos)
        for wp in widget_points:
            painter.fillRect(wp.x() - 4, wp.y() - 4, 8, 8, _ROI_COLOR)

    def _paint_multi(self, painter: QPainter) -> None:
        editing_index = self._multi_drag_index if self._multi_drag_mode not in (None, "new") else -1
        for index, roi in enumerate(self._rois):
            if index == editing_index:
                continue  # canlı sürükleme dikdörtgeni aşağıda ayrıca çizilir
            rect = dt.image_rect_to_widget(roi, self._image_size, self._widget_size())
            self._draw_roi_box(painter, rect, f"ROI{index + 1}")

        if self._multi_live_rect is not None:
            live_index = self._multi_drag_index if self._multi_drag_mode != "new" else len(self._rois)
            self._draw_roi_box(painter, self._multi_live_rect, f"ROI{live_index + 1}")

    def _draw_roi_box(self, painter: QPainter, rect: QRect, label: str) -> None:
        painter.drawRect(rect)
        for hp in self._handle_points(rect).values():
            painter.fillRect(hp.x() - 4, hp.y() - 4, 8, 8, _ROI_COLOR)
        painter.drawText(rect.x() + 2, max(12, rect.y() - 4), label)
