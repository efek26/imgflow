"""`analysis.color_props`/`analysis.texture_props`'un nesne-başına (Otomatik Nesne Tespiti /
Elle ROI Çiz) overlay çizimi için paylaşılan yardımcılar.

İki operatör de daha önce KENDİ `render_*_overlay_multi`'sinde aynı "numarala + sınırlayıcı
kutuyu çiz + altına yaz" kodunu tekrarlıyordu; iki gerçek kullanıcı isteği bu kodu tek bir
yerde toplamayı gerektirdi:

1. **"Şekli çizerken ROI'nin dışına çıkıyor, çıkmasın."** Etiket/değer yazıları kutunun ÜSTÜNE
   (`y - 5`) ve ALTINA (`y + h + satır`) yazılıyordu — yani çizilen ROI'nin/nesnenin DIŞINA
   taşıyor, komşu bölgenin üzerine biniyordu. Artık yazılar sığdığı sürece kutunun İÇİNE
   yerleştirilir (numara üstte, değerler altta); sığmayacak kadar küçük nesnelerde (içeri
   yazmak nesneyi tamamen örterdi) eski "altına yaz" davranışına düşülür ama bu kez görüntü
   sınırlarına KIRPILARAK.
2. **"Numaralandırdığımız kutuyu cismin hemen etrafına kontür gibi çizebiliriz."** Tespit
   edilen nesnenin GERÇEK dış hattı (`contour_from_mask`) varsa düz dikdörtgen yerine o
   çizilir. Elle ROI Çiz modunda kontur YOKTUR (kullanıcının çizdiği dikdörtgenin kendisi
   sınırdır) — orada dikdörtgen çizilmeye devam eder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""`region_props.py`/`shape_matching.py` ile AYNI ölçekleme deseni — bkz. o dosyalardaki
ayrıntılı açıklama."""

_TEXT_PAD_PX = 3
"""Yazının kutu kenarına yapışmaması için bırakılan iç boşluk."""


def contour_from_mask(mask: np.ndarray, offset_x: int, offset_y: int) -> np.ndarray | None:
    """Bir `DetectedObject.mask`'inden (bbox kırpımı boyutunda bool dizi) nesnenin dış
    hattını, TAM GÖRÜNTÜ koordinatlarına ötelenmiş olarak döner.

    Birden fazla dış kontur bulunursa en büyük ALANLI olan seçilir: maske zaten TEK bir bağlı
    bileşenden geldiği için normalde tek kontur olur, ama `fill_holes`/`close_kernel_size`
    gibi morfolojik adımlar sonrası bbox kırpımında komşu bir nesnenin köşesi de kalabilir.
    Kontur bulunamazsa `None` döner — çağıran taraf dikdörtgene düşer."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    return contour + np.array([[[offset_x, offset_y]]], dtype=contour.dtype)


@dataclass
class ObjectOverlayEntry:
    """Tek bir nesne/ROI için çizilecek her şey."""

    measurement: dict[str, Any]
    lines: list[str]
    """Kutunun içine (sığmazsa altına) yazılacak değer satırları."""
    color: tuple[int, int, int]
    contour: np.ndarray | None = None
    """Nesnenin gerçek dış hattı (tam görüntü koordinatında). `None` ise dikdörtgen çizilir."""


def draw_labeled_objects(
    base_image: np.ndarray, entries: list[ObjectOverlayEntry]
) -> np.ndarray:
    """Her nesneyi kendi sınırıyla (kontur ya da dikdörtgen) çizip numaralandırır ve değer
    satırlarını mümkün olduğunca sınırın İÇİNE yazar (bkz. modül docstring'i)."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    img_h, img_w = overlay.shape[:2]
    scale_factor = max(img_h, img_w) / _TEXT_SCALE_REFERENCE_DIM
    box_thickness = max(2, round(2 * scale_factor))
    font_scale = max(0.45, 0.5 * scale_factor)
    thickness = max(1, round(1.5 * scale_factor))
    line_height = max(16, round(18 * scale_factor))

    for index, entry in enumerate(entries, start=1):
        m = entry.measurement
        x, y, w, h = int(m["bbox_x"]), int(m["bbox_y"]), int(m["bbox_w"]), int(m["bbox_h"])
        if entry.contour is not None:
            cv2.drawContours(overlay, [entry.contour], -1, entry.color, box_thickness)
        else:
            cv2.rectangle(overlay, (x, y), (x + w, y + h), entry.color, box_thickness)

        tag = f"ROI{index}" if m.get("manual") else f"#{index}"
        # Yatayda: metin kutunun sol kenarından başlar, ama görüntünün sağ kenarından
        # TAŞACAKSA sola kaydırılır (aksi halde sağdaki nesnelerin değerleri kırpılıyordu).
        text_width = max(
            (
                cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0][0]
                for line in [tag, *entry.lines]
            ),
            default=0,
        )
        text_x = max(0, min(x + _TEXT_PAD_PX, img_w - text_width - _TEXT_PAD_PX))
        # Numara + değer satırları kutunun içine sığıyor mu? (numara 1 satır)
        fits_inside = h >= (len(entry.lines) + 1) * line_height + 2 * _TEXT_PAD_PX
        if fits_inside:
            tag_y = y + line_height
            first_line_y = y + h - _TEXT_PAD_PX - (len(entry.lines) - 1) * line_height
        else:
            # Nesne yazıları içine alamayacak kadar küçük -- içine yazmak nesneyi tamamen
            # örterdi. Eski davranış (üstte numara, altta değerler) korunur ama bu kez
            # görüntü sınırlarına kırpılır.
            tag_y = max(line_height, y - _TEXT_PAD_PX)
            first_line_y = min(y + h + line_height, img_h - _TEXT_PAD_PX)

        cv2.putText(
            overlay, tag, (text_x, tag_y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, entry.color, thickness, cv2.LINE_AA,
        )
        line_y = first_line_y
        for line in entry.lines:
            cv2.putText(
                overlay, line, (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, entry.color, thickness, cv2.LINE_AA,
            )
            line_y += line_height
    return overlay
