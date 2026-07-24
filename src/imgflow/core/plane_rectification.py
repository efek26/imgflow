"""Kamera referans düzleme (banda) tam dik olmasa bile, KONUM-BAĞIMSIZ gerçek-dünya (mm)
ölçüm için düzlem homografisi.

Tek bir `mm_per_px` SKALERİ, kamera düzleme dik değilse (birkaç derecelik bir açı bile)
gerçekte konuma göre DEĞİŞEN bir ölçeği sabit bir sayıyla yaklaşık temsil eder — bu yüzden
aynı cisim ekranın farklı yerlerinde FARKLI ölçülür (gerçek bir kullanıcı raporuyla
doğrulandı: aynı cisim 6mm/5.5mm). Bunun tek kesin çözümü: referans checkerboard karesinin
TAM pozundan (rotasyon + öteleme — `LensCalibrator.rvecs`/`tvecs`) düzlem homografisini
çözüp, ham kareyi bu homografiyle "kuşbakışı" (top-down) bir görünüme çevirmek
(`cv2.warpPerspective`) — o görünümde HER piksel, kamera eğikliği ne olursa olsun, o düzlem
üzerindeki HERHANGİ bir noktada AYNI gerçek mm'i temsil eder (bkz. bu modülün testleri —
sentetik 15°'lik bir kamera eğikliğinde bile kare-kenarı ölçümü karenin merkezinde de
kenarında da tam doğru çıkıyor).

Bu düzeltme SADECE referans düzlemi (checkerboard'ın "Referans (Bant Seviyesi)" olarak
işaretlendiği yakalamadaki konumu) ÜZERİNDEKİ cisimler için geçerlidir — o düzlemden farklı
bir yükseklikteki cisimler için hâlâ hata olur (ayrı bir sorun, bkz. `core/focus_distance.py`
ve `_active_height_mm`)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class PlaneRectification:
    homography: np.ndarray
    """3x3 — ham (lens-undistort edilmiş) kare pikselinden rektifiye çıktı pikseline."""
    output_size: tuple[int, int]
    """(genişlik, yükseklik) — rektifiye çıktı görüntüsünün boyutu."""
    mm_per_px: float
    """Rektifiye görüntüde HER pikselin temsil ettiği SABİT gerçek mm — konuma göre
    DEĞİŞMEZ (naif tek-skaler yaklaşımın aksine), çünkü homografi zaten kamera açısını
    hesaba katarak üretildi."""

    def rectify(self, frame: np.ndarray) -> np.ndarray:
        return cv2.warpPerspective(frame, self.homography, self.output_size)

    def to_dict(self) -> dict[str, Any]:
        return {
            "homography": self.homography.tolist(),
            "output_size": list(self.output_size),
            "mm_per_px": self.mm_per_px,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> PlaneRectification:
        return PlaneRectification(
            homography=np.array(data["homography"], dtype=np.float64),
            output_size=(int(data["output_size"][0]), int(data["output_size"][1])),
            mm_per_px=float(data["mm_per_px"]),
        )


def _apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    transformed = (h @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def compute_plane_rectification(
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    image_size: tuple[int, int],
    mm_per_px: float | None = None,
) -> PlaneRectification:
    """`rvec`/`tvec`, referans checkerboard karesinin `LensCalibrator.rvecs`/`tvecs`'inden
    (o karenin çözülen TAM pozu). `image_size`: (genişlik, yükseklik) — ham/undistort
    edilmiş kare boyutu (aynı boyut, `LensProfile.image_size`).

    `mm_per_px` verilmezse, çıktı çözünürlüğü girdiyle YAKLAŞIK aynı kalacak (gereksiz
    büyütme/çözünürlük kaybı olmasın diye) referans mesafesinden (`fx`) otomatik seçilir.

    Çıktı homografisi, kaynak karenin görebildiği TÜM alanı (4 köşesinin düzlemdeki izdüşümü)
    kapsayacak bir kutuya sığacak şekilde ölçeklenir/kaydırılır — bu yüzden `output_size`
    girdi `image_size`'dan farklı (genelde biraz daha büyük) olabilir."""
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    # Düzlem homografisi (board'un kendi z=0 mm koordinat çerçevesi -> görüntü pikseli):
    # z=0 düzlemindeki noktalar için üçüncü rotasyon sütunu (r3) katkı vermez, bu yüzden
    # standart "Zhang" düzlem homografisi sadece ilk iki rotasyon sütunu + öteleme kullanır.
    r1 = rotation_matrix[:, 0]
    r2 = rotation_matrix[:, 1]
    t = tvec.reshape(3)
    world_to_image = camera_matrix @ np.column_stack([r1, r2, t])
    image_to_world = np.linalg.inv(world_to_image)

    if mm_per_px is None:
        fx = float(camera_matrix[0, 0])
        distance_mm = float(np.linalg.norm(tvec))
        mm_per_px = distance_mm / fx

    width, height = image_size
    corners_px = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float64)
    corners_world = _apply_homography(image_to_world, corners_px)
    min_x, min_y = corners_world.min(axis=0)
    max_x, max_y = corners_world.max(axis=0)

    out_w = max(1, int(round((max_x - min_x) / mm_per_px)))
    out_h = max(1, int(round((max_y - min_y) / mm_per_px)))

    # world (mm) -> rektifiye piksel: ölçekle ve sol-üst köşeyi (min_x, min_y) sıfıra kaydır.
    world_to_output = np.array(
        [
            [1.0 / mm_per_px, 0.0, -min_x / mm_per_px],
            [0.0, 1.0 / mm_per_px, -min_y / mm_per_px],
            [0.0, 0.0, 1.0],
        ]
    )
    homography = world_to_output @ image_to_world

    return PlaneRectification(homography=homography, output_size=(out_w, out_h), mm_per_px=mm_per_px)
