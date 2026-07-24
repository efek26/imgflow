"""Checkerboard'ın kameraya göre 3D pozunu (solvePnP) çözer.

Bu, açılı (dik olmayan) kamera montajında yükseklik-ölçek öğretme noktalarını doğru
hesaplamak için gerekli: kamera dikeyken "kameraya olan mesafe" ile "yükseklik" birbirine
sabit bir ofsetle bağlıdır ve basit bir varsayımla tahmin edilebilir, ama kamera açılıyken
görüntüdeki bir noktanın gerçek 3D mesafesi görüş açısına göre değişir — bunu doğru
hesaplamanın tek yolu, checkerboard'ın (bilinen gerçek boyutlu) köşe noktalarından kameraya
göre TAM 3D pozunu (rotasyon + öteleme) çözmektir. `cv2.solvePnP` bunu, açı ne olursa olsun
doğru şekilde yapar — kullanıcının açıyı elle ölçmesine veya varsaymasına gerek kalmaz.

Bunun için kameranın intrinsic kalibrasyonunun (bkz. `lens_calibration.py`) önceden
tamamlanmış olması ZORUNLU — solvePnP, kamera matrisi/distorsiyon katsayıları olmadan
görüntü noktalarından 3D mesafe çözemez.
"""

from __future__ import annotations

import cv2
import numpy as np

from imgflow.core.lens_calibration import LensProfile


def estimate_distance_mm(
    object_points: np.ndarray, image_points: np.ndarray, lens_profile: LensProfile
) -> float:
    """Checkerboard'ın kameraya olan GERÇEK 3D mesafesini (mm) döner.

    `object_points`: `lens_calibration.build_object_points(...)` ile üretilen, board'un
    kendi düzleminde (z=0) bilinen gerçek köşe koordinatları.
    `image_points`: `lens_calibration.detect_checkerboard_corners(...)`'dan gelen, o karede
    tespit edilmiş köşelerin piksel koordinatları (aynı sırada, object_points ile eşleşmeli).

    Kamera açılı monte edilmiş olsa bile doğru sonuç verir — solvePnP düzlemin tam 3D
    pozunu (rotasyon dahil) çözer, dikey/perpendicular bir varsayım yapmaz.
    """
    ok, _rvec, tvec = cv2.solvePnP(
        object_points, image_points, lens_profile.camera_matrix, lens_profile.dist_coeffs
    )
    if not ok:
        raise ValueError("Checkerboard pozu çözülemedi (solvePnP başarısız).")
    return float(np.linalg.norm(tvec))
