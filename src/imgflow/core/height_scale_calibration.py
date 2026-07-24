"""Yükseklik→ölçek (px/mm) "öğretme" modeli.

Kamera SABİT ve TAM DİKEY monte edilmiş (bant düzlemine dik) varsayımıyla, pinhole kamera
modeli şu ilişkiyi verir: `scale(h) = f_eff / (H_camera - h)` (px/mm), burada `h` ürün
yüksekliği (mm), `H_camera` kameranın bant düzleminden efektif yüksekliği, `f_eff` odak
uzaklığına bağlı bir sabittir. `u = 1/scale` ifadesi `h`'ye göre DOĞRUSALDIR:
`u = H_camera/f_eff - h/f_eff`. Bu yüzden birden fazla (yükseklik, ölçek) "öğretme" noktası
toplanıp `numpy.polyfit` ile bu doğru fit edilerek `f_eff` ve `H_camera` çözülebilir —
ne kadar çok/çeşitli nokta toplanırsa fit o kadar sağlam olur (gürültüye karşı yakınsama).

Öğretme noktası şöyle elde edilir: bilinen bir yükseklikte, bilinen gerçek mesafedeki bir
referans üzerinde iki noktaya tıklanır (bkz. `ui/widgets/measure_canvas.py`); o anki piksel
mesafesi / gerçek mm mesafesi, o yükseklikteki ölçüm ölçeğini (px/mm) doğrudan verir.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TeachPoint:
    height_mm: float
    pixel_distance: float
    real_mm: float

    @property
    def scale_px_per_mm(self) -> float:
        return self.pixel_distance / self.real_mm

    def to_dict(self) -> dict[str, float]:
        return {"height_mm": self.height_mm, "pixel_distance": self.pixel_distance, "real_mm": self.real_mm}

    @staticmethod
    def from_dict(data: dict[str, float]) -> TeachPoint:
        return TeachPoint(
            height_mm=data["height_mm"], pixel_distance=data["pixel_distance"], real_mm=data["real_mm"]
        )


@dataclass
class HeightScaleModel:
    points: list[TeachPoint] = field(default_factory=list)
    f_eff: float | None = None
    h_camera: float | None = None

    def add_point(self, point: TeachPoint) -> None:
        self.points.append(point)

    def fit(self) -> None:
        distinct_heights = {p.height_mm for p in self.points}
        if len(distinct_heights) < 2:
            raise ValueError("En az 2 farklı yükseklikte öğretme noktası gerekli.")

        h = np.array([p.height_mm for p in self.points], dtype=np.float64)
        u = np.array([1.0 / p.scale_px_per_mm for p in self.points], dtype=np.float64)
        slope, intercept = np.polyfit(h, u, 1)
        if slope >= 0:
            raise ValueError(
                "Fit tutarsız (eğim negatif olmalı — yükseklik arttıkça ölçek artmalı). "
                "Öğretme verisini kontrol edin."
            )
        self.f_eff = -1.0 / slope
        self.h_camera = -intercept / slope

    def predict_scale(self, height_mm: float) -> float:
        if self.f_eff is None or self.h_camera is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        denom = self.h_camera - height_mm
        if denom <= 0:
            raise ValueError(
                f"Geçersiz yükseklik ({height_mm}): kamera yüksekliğinden ({self.h_camera}) "
                "büyük veya eşit olamaz."
            )
        return self.f_eff / denom

    def residuals_mm_per_px(self) -> list[float]:
        """Her öğretme noktası için gerçek (1/ölçek) ile modelin öngördüğü (1/ölçek)
        arasındaki fark (mm/px) — fit kalitesi/yakınsama göstergesi. Sıfıra ne kadar
        yakınsa fit o kadar iyi; büyük kalan değerler daha fazla/daha çeşitli öğretme
        noktası eklenmesi gerektiğine işaret eder."""
        if self.f_eff is None or self.h_camera is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        residuals = []
        for point in self.points:
            actual_u = 1.0 / point.scale_px_per_mm
            predicted_u = 1.0 / self.predict_scale(point.height_mm)
            residuals.append(actual_u - predicted_u)
        return residuals

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "f_eff": self.f_eff,
            "h_camera": self.h_camera,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> HeightScaleModel:
        model = HeightScaleModel(points=[TeachPoint.from_dict(p) for p in data.get("points", [])])
        model.f_eff = data.get("f_eff")
        model.h_camera = data.get("h_camera")
        return model
