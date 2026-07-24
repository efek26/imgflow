"""Netlik (odak bulanıklığı) → mesafe (mm) DENEYSEL tahmin modeli.

Kullanıcı hiçbir zaman elle yükseklik/mesafe girmek istemiyor; ürünlerin yanında da
checkerboard olmayacak (bkz. `board_pose.py`/`lens_calibration.py`'deki solvePnP tabanlı
yöntem, sadece checkerboard varken çalışır). Tek kameradan, bilinmeyen boyuttaki bir cismin
GERÇEK mesafesini/yüksekliğini çıkarmanın görüntüdeki görünür boyuttan başka bir yolu yok —
o da döngüsel (bilinmeyen boyutu ölçmek için zaten bilinen boyuta ihtiyaç duyar). Tek
kullanılabilir fizik ipucu: kamera belirli bir çalışma mesafesine odaklanmışsa, o mesafeden
uzaklaştıkça görüntü bulanıklaşır ("depth from defocus").

Bu modül bu ipucunu, checkerboard'ı FARKLI YÜKSEKLİKLERDE göstererek (netlik ölçüsü ↔
`lens_calibration.LensCalibrator.calibrate()`'ın verdiği gerçek `tvec` mesafesi) öğretir.
KESİN/GARANTİLİ bir çözüm DEĞİLDİR — kameranın odak derinliğine (diyafram/lens) bağlı olarak
sinyal güçlü ya da zayıf olabilir; bu yüzden `correlation` (r) değeri de saklanır ve
UI'da "(deneysel)" etiketiyle, zayıfsa açık bir uyarıyla gösterilir (bkz.
`ui/dialogs/lens_calibration_dialog.py`). Tahminler her zaman kalibrasyonda görülen
mesafe aralığına KENETLENİR (`predict_distance`) — zayıf/gürültülü bir fit, aralık dışına
çılgınca savrulan bir tahmin üretmesin diye.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from imgflow.core.focus_metric import focus_measure

_MIN_REQUIRED_POINTS = 2


@dataclass(frozen=True)
class FocusTeachPoint:
    focus_value: float
    distance_mm: float

    def to_dict(self) -> dict[str, float]:
        return {"focus_value": self.focus_value, "distance_mm": self.distance_mm}

    @staticmethod
    def from_dict(data: dict[str, float]) -> FocusTeachPoint:
        return FocusTeachPoint(focus_value=data["focus_value"], distance_mm=data["distance_mm"])


@dataclass
class FocusDistanceModel:
    points: list[FocusTeachPoint] = field(default_factory=list)
    slope: float | None = None
    intercept: float | None = None
    correlation: float | None = None
    """Pearson r (netlik ölçüsü ile mesafe arasında) — fit kalitesinin/sinyal gücünün
    göstergesi. |r| küçükse (ör. <0.4) bu model pratikte GÜVENİLMEMELİDİR; kamera muhtemelen
    bu yükseklik aralığında anlamlı bir odak/bulanıklık farkı üretmiyordur (derin odak)."""

    def add_point(self, point: FocusTeachPoint) -> None:
        self.points.append(point)

    def fit(self) -> None:
        if len(self.points) < _MIN_REQUIRED_POINTS:
            raise ValueError(f"En az {_MIN_REQUIRED_POINTS} öğretme noktası gerekli (şu an {len(self.points)}).")

        focus_values = np.array([p.focus_value for p in self.points], dtype=np.float64)
        distances = np.array([p.distance_mm for p in self.points], dtype=np.float64)
        if np.allclose(focus_values, focus_values[0]):
            raise ValueError(
                "Tüm karelerin netlik ölçüsü aynı görünüyor — kamerada bu yükseklik "
                "aralığında algılanabilir bir odak/bulanıklık farkı yok."
            )

        slope, intercept = np.polyfit(focus_values, distances, 1)
        self.slope = float(slope)
        self.intercept = float(intercept)
        self.correlation = float(np.corrcoef(focus_values, distances)[0, 1])

    def predict_distance(self, focus_value: float) -> float:
        if self.slope is None or self.intercept is None:
            raise RuntimeError("Önce fit() çağrılmalı.")
        predicted = self.slope * focus_value + self.intercept
        calibrated_distances = [p.distance_mm for p in self.points]
        return float(np.clip(predicted, min(calibrated_distances), max(calibrated_distances)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "points": [p.to_dict() for p in self.points],
            "slope": self.slope,
            "intercept": self.intercept,
            "correlation": self.correlation,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> FocusDistanceModel:
        model = FocusDistanceModel(points=[FocusTeachPoint.from_dict(p) for p in data.get("points", [])])
        model.slope = data.get("slope")
        model.intercept = data.get("intercept")
        model.correlation = data.get("correlation")
        return model


def model_from_calibration_frames(frames: list[np.ndarray], tvecs: list[np.ndarray]) -> FocusDistanceModel:
    """`frames[i]`'in netlik ölçüsünü, `tvecs[i]`'den çözülen gerçek mesafeyle eşleştirip
    DENEYSEL bir netlik→mesafe modeli fit eder. `frames` ve `tvecs` AYNI SIRADA olmalı
    (`LensCalibrationDialog._build_calibrator_from_selection`'ın döndürdüğü, kalibrasyona
    fiilen dahil edilmiş ham kareler ile `LensCalibrator.tvecs`)."""
    model = FocusDistanceModel()
    for frame, tvec in zip(frames, tvecs):
        distance = float(np.linalg.norm(tvec))
        model.add_point(FocusTeachPoint(focus_value=focus_measure(frame), distance_mm=distance))
    return model
