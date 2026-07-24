"""Diskten görüntü okuma/yazma. imgflow genelinde OpenCV'nin BGR kanal sırası kullanılır.

cv2.imread/imwrite Windows'ta Unicode (ör. Türkçe karakter içeren "Masaüstü" gibi) yolları
güvenilir açamıyor; bu yüzden dosya I/O'sunu Path.read_bytes/write_bytes ile yapıp
içeriği cv2.imdecode/imencode ile çözüyoruz.

Tüm operatörler (ve UI'ın numpy_to_qimage'ı) yalnızca gri (2B) veya 3 kanallı BGR
görüntü bekliyor; alfa kanallı (BGRA/RGBA, ör. şeffaf PNG) dosyalar burada BGR'a
düşürülür ki pipeline'ın geri kalanı 4. kanalla hiç karşılaşmasın.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}") from exc

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {path}")
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buffer = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise OSError(f"Görüntü kaydedilemedi: {path}")
    path.write_bytes(buffer.tobytes())
