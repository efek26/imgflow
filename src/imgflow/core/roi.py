"""ROI (ilgi alanı) veri modeli: eksene hizalı dikdörtgen.

v1 kapsamı basit dikdörtgenle sınırlı; FAZ2'deki geometrik eşleme (x, y, alpha) çıktısı
ile döndürülmüş/serbest biçimli bölgeler ayrı bir tipte ele alınacak.
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
