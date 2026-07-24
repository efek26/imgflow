import cv2
import numpy as np
import pytest

from imgflow.core.plane_rectification import PlaneRectification, compute_plane_rectification

_K = np.array([[1000.0, 0.0, 640.0], [0.0, 1000.0, 480.0], [0.0, 0.0, 1.0]])
_IMAGE_SIZE = (1280, 960)


def _tilted_pose(angle_deg: float = 15.0, distance_mm: float = 800.0):
    angle = np.deg2rad(angle_deg)
    rvec = np.array([[angle], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [distance_mm]])
    return rvec, tvec


def _apply_homography(h: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1))
    homogeneous = np.hstack([points, ones])
    transformed = (h @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2:3]


def _measure_rect_in_rectified_frame(
    rectification: PlaneRectification, rvec, tvec, world_center_xy, w_mm=50.0, h_mm=30.0
):
    """Dünyadaki (world) bilinen boyutlu bir dikdörtgeni kameranın gördüğü ham piksellere
    projekte eder (`cv2.projectPoints`, gerçek bir kameranın o cismi nasıl gördüğünü taklit
    eder), sonra rektifikasyon homografisinden geçirip rektifiye görüntüdeki ölçümü verir."""
    cx, cy = world_center_xy
    corners_world = np.array(
        [
            [cx - w_mm / 2, cy - h_mm / 2, 0.0],
            [cx + w_mm / 2, cy - h_mm / 2, 0.0],
            [cx + w_mm / 2, cy + h_mm / 2, 0.0],
            [cx - w_mm / 2, cy + h_mm / 2, 0.0],
        ]
    )
    image_points, _ = cv2.projectPoints(corners_world, rvec, tvec, _K, None)
    image_points = image_points.reshape(-1, 2)
    rectified_points = _apply_homography(rectification.homography, image_points)
    measured_w = np.linalg.norm(rectified_points[1] - rectified_points[0]) * rectification.mm_per_px
    measured_h = np.linalg.norm(rectified_points[2] - rectified_points[1]) * rectification.mm_per_px
    return measured_w, measured_h


def test_rectification_gives_exact_size_under_tilted_camera_regardless_of_position():
    """Kamera düzleme 15° eğik olsa bile (gerçek bir montaj hatasını taklit eder), bilinen
    50x30mm'lik bir dikdörtgen karenin MERKEZİNDE de KENARINDA da aynı doğrulukla
    ölçülmeli — bu, tek bir mm_per_px skalerinin ÇÖZEMEYECEĞİ konum-bağımlı hatayı homografi
    ile ortadan kaldırdığımızı kanıtlar."""
    rvec, tvec = _tilted_pose()
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE)

    for center in [(0.0, 0.0), (300.0, 200.0), (-300.0, -200.0)]:
        measured_w, measured_h = _measure_rect_in_rectified_frame(rectification, rvec, tvec, center)
        assert measured_w == pytest.approx(50.0, abs=0.05)
        assert measured_h == pytest.approx(30.0, abs=0.05)


def test_rectification_matches_naive_scalar_when_camera_is_perfectly_perpendicular():
    """Kamera tam dikse (eğim yok), homografi tabanlı ölçüm ile basit tek-skaler yaklaşım
    zaten aynı sonucu vermeli — homografi sadece eğik durumda fark yaratır, dik durumda
    gereksiz bir karmaşıklık/fark getirmemeli."""
    rvec = np.array([[0.0], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [800.0]])
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE)

    measured_w, measured_h = _measure_rect_in_rectified_frame(rectification, rvec, tvec, (0.0, 0.0))
    assert measured_w == pytest.approx(50.0, abs=0.05)
    assert measured_h == pytest.approx(30.0, abs=0.05)


def test_mm_per_px_defaults_close_to_naive_distance_over_fx():
    rvec, tvec = _tilted_pose()
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE)

    naive_mm_per_px = float(np.linalg.norm(tvec)) / float(_K[0, 0])
    assert rectification.mm_per_px == pytest.approx(naive_mm_per_px)


def test_explicit_mm_per_px_is_respected():
    rvec, tvec = _tilted_pose()
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE, mm_per_px=0.5)

    assert rectification.mm_per_px == pytest.approx(0.5)


def test_rectify_produces_expected_output_shape():
    rvec, tvec = _tilted_pose()
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE)
    frame = np.zeros((_IMAGE_SIZE[1], _IMAGE_SIZE[0], 3), dtype=np.uint8)

    rectified = rectification.rectify(frame)

    assert rectified.shape[:2] == (rectification.output_size[1], rectification.output_size[0])


def test_round_trips_through_dict():
    rvec, tvec = _tilted_pose()
    rectification = compute_plane_rectification(_K, rvec, tvec, _IMAGE_SIZE)

    restored = PlaneRectification.from_dict(rectification.to_dict())

    assert np.allclose(restored.homography, rectification.homography)
    assert restored.output_size == rectification.output_size
    assert restored.mm_per_px == pytest.approx(rectification.mm_per_px)
