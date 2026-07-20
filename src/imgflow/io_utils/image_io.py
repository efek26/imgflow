"""Diskten görüntü okuma/yazma. imgflow genelinde OpenCV'nin BGR kanal sırası kullanılır."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise OSError(f"Görüntü kaydedilemedi: {path}")
