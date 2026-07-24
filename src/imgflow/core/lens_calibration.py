"""Lens (intrinsic) kalibrasyon: checkerboard ile distorsiyon düzeltmesi.

Kamera karesinin farklı X/Y bölgelerinde geometrik ölçeğin tutarlı olması için gerekli
(lens distorsiyonu düzeltilmeden karenin kenarlarındaki ölçüm hataları merkeze göre farklı
olur). Kalibre edildikten sonra HER karede otomatik uygulanır (`main_window._on_camera_tick`)
— opsiyonel bir pipeline operatörü DEĞİL, Unicode-safe I/O gibi her zaman doğru olması
gereken bir düzeltme.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

_MIN_RECOMMENDED_FRAMES = 10
_MIN_REQUIRED_FRAMES = 2
_MIN_CHARUCO_CORNERS = 4
"""ChArUco poz/kalibrasyon çözümü için pratikte gereken minimum köşe sayısı (solvePnP'nin
düzlemsel durumda güvenilir çalışması için)."""

_ARUCO_DICTIONARIES: dict[str, int] = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}


@dataclass(frozen=True)
class CheckerboardConfig:
    inner_cols: int
    """Satranç tahtasının bir satırındaki İÇ köşe sayısı (kare sayısı değil)."""
    inner_rows: int
    square_size_mm: float


@dataclass(frozen=True)
class CharucoBoardConfig:
    """ChArUco tahtası (checkerboard + benzersiz ID'li ArUco marker'lar) — düz checkerboard'ın
    aksine KISMİ görünürlükte/kırpılmada bile poz/kalibrasyon çözülebilir, çünkü her kare kendi
    ID'siyle tanınır (tüm iç köşelerin görünmesi şart değildir). Dar konveyör bantlarında veya
    yansıma/gölge nedeniyle checkerboard'ın bir kısmı kaybolduğunda bu, düz checkerboard'a göre
    çok daha sağlam bir alternatiftir."""

    squares_x: int
    squares_y: int
    square_length_mm: float
    marker_length_mm: float
    dictionary_name: str = "DICT_4X4_50"


@dataclass
class LensProfile:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: tuple[int, int]  # (genişlik, yükseklik)
    rms_error: float
    _new_camera_matrix: np.ndarray | None = field(default=None, repr=False, compare=False)
    _map1: np.ndarray | None = field(default=None, repr=False, compare=False)
    _map2: np.ndarray | None = field(default=None, repr=False, compare=False)

    def _ensure_maps(self) -> None:
        if self._map1 is not None:
            return
        new_camera_matrix, _roi = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, self.image_size, alpha=0
        )
        map1, map2 = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            None,
            new_camera_matrix,
            self.image_size,
            cv2.CV_16SC2,
        )
        self._new_camera_matrix = new_camera_matrix
        self._map1 = map1
        self._map2 = map2

    def to_dict(self) -> dict[str, Any]:
        return {
            "camera_matrix": self.camera_matrix.tolist(),
            "dist_coeffs": self.dist_coeffs.tolist(),
            "image_size": list(self.image_size),
            "rms_error": self.rms_error,
        }

    @staticmethod
    def from_dict(data: dict[str, Any]) -> LensProfile:
        return LensProfile(
            camera_matrix=np.array(data["camera_matrix"], dtype=np.float64),
            dist_coeffs=np.array(data["dist_coeffs"], dtype=np.float64),
            image_size=(int(data["image_size"][0]), int(data["image_size"][1])),
            rms_error=float(data["rms_error"]),
        )


def undistort(frame: np.ndarray, profile: LensProfile) -> np.ndarray:
    """`profile`'ın undistort map'lerini bir kez hesaplayıp `cv2.remap` ile ucuza uygular
    (her kamera tick'inde çağrılacağı için map yeniden hesaplamak pahalı olurdu)."""
    profile._ensure_maps()
    return cv2.remap(frame, profile._map1, profile._map2, interpolation=cv2.INTER_LINEAR)


def build_object_points(inner_cols: int, inner_rows: int, square_size_mm: float) -> np.ndarray:
    """Bir tek satranç tahtası karesinin 3D nesne noktaları (z=0 düzleminde, board'un kendi
    koordinat çerçevesinde) — `cv2.calibrateCamera`/`cv2.solvePnP` için ortak girdi.
    `LensCalibrator` bunu her kare için aynı şekilde kullanır; `board_pose.py`'deki
    solvePnP tabanlı mesafe kestirimi de aynı fonksiyonu paylaşır (kod tekrarını önler)."""
    objp = np.zeros((inner_cols * inner_rows, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
    objp *= square_size_mm
    return objp


_QUICK_CHECK_MAX_DIM = 400
"""`quick_checkerboard_found`'un küçülttüğü maksimum kenar (px). `cv2.findChessboardCorners`
tahta BULUNAMADIĞINDA (tüm eşikleme seviyelerini/aday dörtgenleri dener) gerçek kamera
çözünürlüğünde (ör. 1280x1024) SANİYELER sürebilir — küçük bir kopyada aynı arama saniyenin
onda biri kadar sürer, gerçek bir checkerboard'ın tespiti ise zaten neredeyse anında olduğu
için küçültmeden etkilenmez (bkz. bu değerlerin ölçüldüğü benchmark)."""


def quick_checkerboard_found(frame: np.ndarray, pattern_size: tuple[int, int]) -> bool:
    """`Kare Yakala` anında (kullanıcı her tıklamada bekliyor) SADECE hızlı bir "bulundu mu"
    önizlemesi için — `detect_checkerboard_corners`'ın SB fallback'ı ve alt-piksel iyileştirmesi
    YOKTUR, bu yüzden kalibrasyon için KULLANILMAMALI. Gerçek/otoriter köşe tespiti (SB fallback
    dahil, tam çözünürlükte) `Kalibre Et` sırasında `detect_checkerboard_corners` ile ayrıca ve
    tam olarak yapılır (bkz. `ui/dialogs/lens_calibration_dialog.py`)."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    scale = _QUICK_CHECK_MAX_DIM / max(h, w)
    if scale < 1.0:
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    found, _corners = cv2.findChessboardCorners(gray, pattern_size)
    return found


def detect_checkerboard_corners(
    frame: np.ndarray, pattern_size: tuple[int, int]
) -> tuple[bool, np.ndarray | None, np.ndarray]:
    """Durum tutmayan (stateless) satranç tahtası köşe tespiti — hem `LensCalibrator.add_frame`
    hem de yükseklik-ölçek öğretme dialogundaki tekil "satranç tahtası ile ekle" akışı
    tarafından paylaşılır. Bulunursa alt-piksel hassasiyetinde iyileştirilmiş köşeleri ve
    işaretli bir önizleme döner; bulunamazsa `(False, None, orijinal_kare)`."""
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    preview = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        # Klasik dedektör aydınlatma/bulanıklık/aşırı perspektifte sık "bulunamadı" verir;
        # OpenCV'nin daha yeni "sector based" dedektörü aynı koşullarda çok daha sağlamdır
        # ve köşeleri zaten alt-piksel hassasiyetinde döner (cornerSubPix gerekmez).
        found, corners = cv2.findChessboardCornersSB(
            gray, pattern_size, flags=cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY
        )

    if not found:
        return False, None, preview

    cv2.drawChessboardCorners(preview, pattern_size, corners, found)
    return True, corners, preview


def build_charuco_board(config: CharucoBoardConfig) -> cv2.aruco.CharucoBoard:
    if config.dictionary_name not in _ARUCO_DICTIONARIES:
        raise ValueError(f"Bilinmeyen ArUco sözlüğü: {config.dictionary_name}")
    dictionary = cv2.aruco.getPredefinedDictionary(_ARUCO_DICTIONARIES[config.dictionary_name])
    return cv2.aruco.CharucoBoard(
        (config.squares_x, config.squares_y), config.square_length_mm, config.marker_length_mm, dictionary
    )


def detect_charuco_corners(
    frame: np.ndarray, board: cv2.aruco.CharucoBoard
) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray]:
    """Durum tutmayan ChArUco köşe tespiti — `detect_checkerboard_corners`'ın ChArUco karşılığı.
    Bulunursa (checkerboard köşesi + eşleşen ArUco ID'si çiftleri) `(True, charuco_corners,
    charuco_ids, preview)`, yeterli köşe bulunamazsa `(False, None, None, orijinal_kare)` döner.
    """
    gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    preview = frame.copy() if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if charuco_corners is None or len(charuco_corners) < _MIN_CHARUCO_CORNERS:
        return False, None, None, preview

    if marker_corners is not None and len(marker_corners) > 0:
        cv2.aruco.drawDetectedMarkers(preview, marker_corners, marker_ids)
    # drawDetectedCornersCharuco 2 kanallı (N,1,2) Mat bekliyor; detectBoard/matchImagePoints
    # ise düz (N,2) döndürüyor — sadece çizim için yeniden şekillendiriyoruz.
    cv2.aruco.drawDetectedCornersCharuco(preview, charuco_corners.reshape(-1, 1, 2), charuco_ids)
    return True, charuco_corners, charuco_ids, preview


class LensCalibrator:
    def __init__(self, config: CheckerboardConfig | None = None) -> None:
        """`config` verilmezse (`add_frame` değil) sadece `add_charuco_frame` kullanılabilir —
        ChArUco'nun kendi board tanımı zaten pattern/object-point bilgisini taşır, ayrıca bir
        `CheckerboardConfig`'e ihtiyaç duymaz."""
        if config is not None:
            self._pattern_size: tuple[int, int] | None = (config.inner_cols, config.inner_rows)
            self._objp: np.ndarray | None = build_object_points(
                config.inner_cols, config.inner_rows, config.square_size_mm
            )
        else:
            self._pattern_size = None
            self._objp = None
        self._object_points: list[np.ndarray] = []
        self._image_points: list[np.ndarray] = []
        self._image_size: tuple[int, int] | None = None
        self._tvecs: list[np.ndarray] | None = None
        self._rvecs: list[np.ndarray] | None = None

    @property
    def captured_count(self) -> int:
        return len(self._image_points)

    @property
    def tvecs(self) -> list[np.ndarray] | None:
        """`calibrate()` sonrası her karenin çözülen 3D öteleme vektörü — sırası
        `add_frame`/`add_charuco_frame` ile eklenme sırasıyla AYNIDIR. `calibrate()`
        çağrılmadan `None`. `np.linalg.norm(tvec)`, o karede checkerboard'ın kameraya olan
        gerçek mesafesini (mm) verir — `board_pose.estimate_distance_mm`'in tek-kareye
        `solvePnP` uyguladığı hesabın aynısı, ama tüm kareler için tek bir bundle-adjustment
        (`cv2.calibrateCamera`) içinde bedava çıkar."""
        return self._tvecs

    @property
    def rvecs(self) -> list[np.ndarray] | None:
        """`tvecs` ile AYNI SIRADA, her karenin çözülen 3D rotasyon vektörü (Rodrigues).
        Tek başına mesafe için gerekmez, ama düzlem homografisi (bkz.
        `core/plane_rectification.py` — kamera referans düzleme tam dik olmasa bile konum-
        bağımsız doğru ölçüm için) TAM pozu (rotasyon + öteleme) gerektirir."""
        return self._rvecs

    def add_frame(self, frame: np.ndarray) -> tuple[bool, np.ndarray]:
        """Karede satranç tahtası köşelerini arar. Bulunursa alt-piksel hassasiyetinde
        iyileştirilip veri setine eklenir. Her durumda köşeleri işaretli bir önizleme döner
        (kullanıcı geri bildirimi için)."""
        if self._pattern_size is None:
            raise RuntimeError(
                "add_frame() için LensCalibrator bir CheckerboardConfig ile oluşturulmalı "
                "(ChArUco kullanıyorsanız add_charuco_frame() çağırın)."
            )
        found, corners, preview = detect_checkerboard_corners(frame, self._pattern_size)
        if not found:
            return False, preview

        self._object_points.append(self._objp.copy())
        self._image_points.append(corners)
        self._image_size = (frame.shape[1], frame.shape[0])
        return True, preview

    def add_charuco_frame(self, frame: np.ndarray, board: cv2.aruco.CharucoBoard) -> tuple[bool, np.ndarray]:
        """`add_frame`'in ChArUco karşılığı — aynı `_object_points`/`_image_points` listelerine
        ekler, bu yüzden `calibrate()` DEĞİŞMEDEN çalışır (checkerboard ve ChArUco kareleri
        aynı `LensCalibrator` örneğinde bile karıştırılabilir, `cv2.calibrateCamera` her kare
        için ayrı sayıda nokta kabul eder)."""
        found, charuco_corners, charuco_ids, preview = detect_charuco_corners(frame, board)
        if not found:
            return False, preview

        obj_points, img_points = board.matchImagePoints(charuco_corners, charuco_ids)
        self._object_points.append(obj_points)
        self._image_points.append(img_points)
        self._image_size = (frame.shape[1], frame.shape[0])
        return True, preview

    def calibrate(self, force: bool = False) -> LensProfile:
        if self.captured_count < _MIN_REQUIRED_FRAMES:
            raise ValueError(
                f"Kalibrasyon için en az {_MIN_REQUIRED_FRAMES} geçerli kare gerekli "
                f"(şu an {self.captured_count})."
            )
        if self.captured_count < _MIN_RECOMMENDED_FRAMES and not force:
            raise ValueError(
                f"En az {_MIN_RECOMMENDED_FRAMES} kare önerilir (şu an {self.captured_count}); "
                "daha güvenilir bir sonuç için farklı açılardan daha fazla kare ekleyin."
            )

        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self._object_points, self._image_points, self._image_size, None, None
        )
        self._tvecs = list(tvecs)
        self._rvecs = list(rvecs)
        return LensProfile(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=self._image_size,
            rms_error=float(rms),
        )
