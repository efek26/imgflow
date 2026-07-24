"""Görüntü <-> widget koordinat dönüşümü için saf (Qt widget'a bağımlı olmayan) fonksiyonlar.

Görüntü, en-boy oranı korunarak ortalanmış şekilde ölçekli gösterildiği için (bkz.
`ImageView`), fare tıklamalarını gerçek görüntü piksel koordinatına çevirmek için bu
dönüşüm gerekir. Bu mantık başlangıçta `RoiCanvas` içinde gömülüydü; `MeasureCanvas` da
aynı dönüşüme ihtiyaç duyduğu için buraya çıkarıldı — her iki widget de bu fonksiyonları
kullanır, kod tekrarı önlenir. Qt widget nesnelerine bağımlı olmadığı için qtbot olmadan
da test edilebilir.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect

Size = tuple[int, int]


def display_rect(image_size: Size, widget_size: Size) -> QRect:
    """Ölçeklenmiş görüntünün widget içindeki (ortalanmış) dikdörtgeni."""
    img_w, img_h = image_size
    if img_w <= 0 or img_h <= 0:
        return QRect()
    label_w, label_h = widget_size
    scale = min(label_w / img_w, label_h / img_h)
    disp_w, disp_h = int(img_w * scale), int(img_h * scale)
    x0, y0 = (label_w - disp_w) // 2, (label_h - disp_h) // 2
    return QRect(x0, y0, disp_w, disp_h)


def display_scale(image_size: Size, widget_size: Size) -> float:
    img_w, _img_h = image_size
    disp = display_rect(image_size, widget_size)
    if img_w <= 0 or disp.isEmpty():
        return 1.0
    return disp.width() / img_w


def widget_point_to_image(point: QPoint, image_size: Size, widget_size: Size) -> tuple[int, int]:
    disp = display_rect(image_size, widget_size)
    img_w, img_h = image_size
    if disp.width() <= 0 or disp.height() <= 0 or img_w <= 0 or img_h <= 0:
        return (0, 0)
    sx, sy = img_w / disp.width(), img_h / disp.height()
    x = int((point.x() - disp.x()) * sx)
    y = int((point.y() - disp.y()) * sy)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    return (x, y)


def image_point_to_widget(point: tuple[int, int], image_size: Size, widget_size: Size) -> QPoint:
    disp = display_rect(image_size, widget_size)
    scale = display_scale(image_size, widget_size)
    x, y = point
    return QPoint(int(disp.x() + x * scale), int(disp.y() + y * scale))


def widget_rect_to_image(rect: QRect, image_size: Size, widget_size: Size) -> tuple[int, int, int, int]:
    disp = display_rect(image_size, widget_size)
    img_w, img_h = image_size
    if disp.width() <= 0 or disp.height() <= 0 or img_w <= 0 or img_h <= 0:
        return (0, 0, 0, 0)
    sx, sy = img_w / disp.width(), img_h / disp.height()
    x = int((rect.x() - disp.x()) * sx)
    y = int((rect.y() - disp.y()) * sy)
    w = int(rect.width() * sx)
    h = int(rect.height() * sy)
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return (x, y, w, h)


def image_rect_to_widget(rect: tuple[int, int, int, int], image_size: Size, widget_size: Size) -> QRect:
    disp = display_rect(image_size, widget_size)
    img_w, img_h = image_size
    if disp.isEmpty() or img_w <= 0 or img_h <= 0:
        return QRect()
    sx, sy = disp.width() / img_w, disp.height() / img_h
    x, y, w, h = rect
    return QRect(int(disp.x() + x * sx), int(disp.y() + y * sy), int(w * sx), int(h * sy))


def widget_circle_to_image(
    center: QPoint, radius: int, image_size: Size, widget_size: Size
) -> tuple[int, int, int]:
    disp = display_rect(image_size, widget_size)
    img_w, img_h = image_size
    if disp.width() <= 0 or img_w <= 0 or img_h <= 0:
        return (0, 0, 0)
    scale = img_w / disp.width()
    cx = int((center.x() - disp.x()) * scale)
    cy = int((center.y() - disp.y()) * scale)
    r = max(1, int(radius * scale))
    cx = max(0, min(cx, img_w))
    cy = max(0, min(cy, img_h))
    return (cx, cy, r)


def image_circle_to_widget(
    cx: int, cy: int, r: int, image_size: Size, widget_size: Size
) -> tuple[QPoint, int] | None:
    disp = display_rect(image_size, widget_size)
    if disp.isEmpty():
        return None
    scale = display_scale(image_size, widget_size)
    center = QPoint(int(disp.x() + cx * scale), int(disp.y() + cy * scale))
    return center, int(r * scale)
