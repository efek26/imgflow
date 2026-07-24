"""ROI (ilgi alanı) veri modelleri: eksene hizalı dikdörtgen ve daire.

v1 kapsamı dikdörtgen + daire ile sınırlı; FAZ2'deki geometrik eşleme (x, y, alpha) çıktısı
ile döndürülmüş/serbest biçimli (poligon) bölgeler ayrı bir tipte ele alınacak.
"""

from __future__ import annotations

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
