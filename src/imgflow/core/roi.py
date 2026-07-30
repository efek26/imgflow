"""ROI (ilgi alanı) veri modelleri: eksene hizalı dikdörtgen ve daire.

v1 kapsamı dikdörtgen + daire ile sınırlı; FAZ2'deki geometrik eşleme (x, y, alpha) çıktısı
ile döndürülmüş/serbest biçimli (poligon) bölgeler ayrı bir tipte ele alınacak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class RoiRect:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, image_w: int, image_h: int) -> RoiRect:
        x = max(0, min(self.x, image_w))
        y = max(0, min(self.y, image_h))
        w = max(0, min(self.w, image_w - x))
        h = max(0, min(self.h, image_h - y))
        return RoiRect(x, y, w, h)

    def as_slice(self) -> tuple[slice, slice]:
        return slice(self.y, self.y + self.h), slice(self.x, self.x + self.w)


@dataclass(frozen=True)
class RoiCircle:
    cx: int
    cy: int
    r: int

    def bounding_rect(self) -> RoiRect:
        return RoiRect(self.cx - self.r, self.cy - self.r, 2 * self.r, 2 * self.r)

    def clamp(self, image_w: int, image_h: int) -> RoiCircle:
        """Dairenin tamamı görüntü sınırları içinde kalacak şekilde kırpar.

        Kısmen görüntü dışına taşan daireler yerine merkezi/yarıçapı sınırlara çekilir;
        böylece maske her zaman tam bir daire olur, kenarda kesik yarım daire oluşmaz.
        """
        max_r = max(1, min(image_w, image_h) // 2)
        r = max(1, min(self.r, max_r))
        cx = max(r, min(self.cx, image_w - r))
        cy = max(r, min(self.cy, image_h - r))
        return RoiCircle(cx, cy, r)


def parse_roi_list(data: str, image_w: int, image_h: int) -> list[RoiRect]:
    """`analysis.color_props`/`analysis.texture_props`'un "Elle ROI Çiz" modu için:
    `RoiCanvas`'ta fare ile çizilen dikdörtgenlerin JSON-kodlanmış listesini
    (`"[[x,y,w,h], ...]"`) `RoiRect`'e çevirir, her birini görüntü sınırlarına kırpar,
    bozuk/boş girdileri sessizce eler (canvas zaten geçerli dikdörtgenler üretir, ama
    JSON elle bozulursa -- ör. reçete dosyası elle düzenlenirse -- pipeline çökmemeli)."""
    try:
        raw = json.loads(data)
    except (ValueError, TypeError):
        return []
    if not isinstance(raw, list):
        return []

    rois: list[RoiRect] = []
    for item in raw:
        try:
            x, y, w, h = (int(v) for v in item)
        except (TypeError, ValueError):
            continue
        rect = RoiRect(x, y, w, h).clamp(image_w, image_h)
        if rect.w > 0 and rect.h > 0:
            rois.append(rect)
    return rois
