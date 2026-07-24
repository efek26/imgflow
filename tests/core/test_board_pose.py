import cv2
import numpy as np
import pytest

from imgflow.core.board_pose import estimate_distance_mm
from imgflow.core.lens_calibration import CharucoBoardConfig, LensProfile, build_charuco_board, build_object_points

_CAMERA_MATRIX = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
_DIST_COEFFS = np.zeros((5, 1))
_LENS_PROFILE = LensProfile(
    camera_matrix=_CAMERA_MATRIX, dist_coeffs=_DIST_COEFFS, image_size=(640, 480), rms_error=0.1
)


def _synthetic_image_points(object_points: np.ndarray, rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    image_points, _ = cv2.projectPoints(object_points, rvec, tvec, _CAMERA_MATRIX, _DIST_COEFFS)
    return image_points.reshape(-1, 2)


def test_estimate_distance_mm_recovers_known_distance_facing_camera():
    object_points = build_object_points(inner_cols=5, inner_rows=4, square_size_mm=25.0)
    true_distance = 500.0
    rvec = np.zeros((3, 1))
    tvec = np.array([[0.0], [0.0], [true_distance]])
    image_points = _synthetic_image_points(object_points, rvec, tvec)

    distance = estimate_distance_mm(object_points, image_points, _LENS_PROFILE)

    assert distance == pytest.approx(true_distance, rel=1e-3)


def test_estimate_distance_mm_works_with_charuco_matched_points():
    """ChArUco'nun `board.matchImagePoints(...)` çıktısı ile AYNI şekil/tipte (N,3)/(N,2)
    float32 noktalar üretir (`board.getChessboardCorners()`); `estimate_distance_mm`'in
    checkerboard'a özel bir varsayım yapmadan, herhangi bir eşleşmiş nokta kaynağıyla
    çalıştığını doğrular — ChArUco akışı için gerçek görüntü tespiti gerekmez."""
    board = build_charuco_board(
        CharucoBoardConfig(squares_x=5, squares_y=4, square_length_mm=25.0, marker_length_mm=18.0)
    )
    object_points = board.getChessboardCorners()
    rvec = np.array([[0.1], [0.15], [0.05]])
    tvec = np.array([[10.0], [5.0], [350.0]])
    true_distance = float(np.linalg.norm(tvec))
    image_points = _synthetic_image_points(object_points, rvec, tvec)

    distance = estimate_distance_mm(object_points, image_points, _LENS_PROFILE)

    assert distance == pytest.approx(true_distance, rel=1e-3)


def test_estimate_distance_mm_correct_when_board_is_tilted_and_offset():
    """Kamera/board açılı olsa bile (dikey varsayım YAPMADAN) doğru 3D mesafeyi vermeli —
    açılı kamera montajı desteğinin tüm amacı bu."""
    object_points = build_object_points(inner_cols=5, inner_rows=4, square_size_mm=25.0)
    rvec = np.array([[0.3], [0.2], [0.1]])  # rastgele bir eğim/döndürme
    tvec = np.array([[40.0], [-20.0], [420.0]])
    true_distance = float(np.linalg.norm(tvec))
    image_points = _synthetic_image_points(object_points, rvec, tvec)

    distance = estimate_distance_mm(object_points, image_points, _LENS_PROFILE)

    assert distance == pytest.approx(true_distance, rel=1e-3)
