"""Netlik (odak) ölçüsü — "variance of Laplacian", standart bir makine görüşü odak metriği.

Net görüntüde (keskin kenarlar) yüksek, bulanık görüntüde düşük değer verir. Deneysel
odak-tabanlı yükseklik tahmini (`core/focus_distance.py`) bu metriği hem kalibrasyon
sırasında (checkerboard'ın farklı yüksekliklerdeki netliği) hem çalışma zamanında (gerçek
cismin netliği) aynı şekilde kullanır.
"""

from __future__ import annotations

import cv2
import numpy as np


def focus_measure(frame: np.ndarray) -> float:
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())
