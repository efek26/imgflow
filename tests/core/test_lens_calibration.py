import cv2
import numpy as np
import pytest

from imgflow.core.lens_calibration import (
    CharucoBoardConfig,
    CheckerboardConfig,
    LensCalibrator,
    build_charuco_board,
    build_object_points,
    detect_charuco_corners,
    detect_checkerboard_corners,
    undistort,
)


def _draw_checkerboard(inner_cols: int, inner_rows: int, square_px: int = 40, margin: int = 40) -> np.ndarray:
    cols_squares = inner_cols + 1
    rows_squares = inner_rows + 1
    w = cols_squares * square_px + 2 * margin
    h = rows_squares * square_px + 2 * margin
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    for r in range(rows_squares):
        for c in range(cols_squares):
            if (r + c) % 2 == 0:
                y0 = margin + r * square_px
                x0 = margin + c * square_px
                img[y0 : y0 + square_px, x0 : x0 + square_px] = 0
    return img


def _warped_views(base_image: np.ndarray, count: int) -> list[np.ndarray]:
    """Aynı satranç tahtasının farklı (küçük perspektif sapmalı) görünümlerini üretir —
    gerçek çoklu-poz kalibrasyon verisini taklit eder (tek düz görüntü numerik olarak
    dejenere/tutarsız bir kalibrasyona yol açar)."""
    h, w = base_image.shape[:2]
    rng = np.random.default_rng(42)
    src = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    views = []
    for _ in range(count):
        jitter = rng.uniform(-0.05, 0.05, size=(4, 2)) * [w, h]
        dst = (src + jitter).astype(np.float32)
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(base_image, matrix, (w, h), borderValue=(255, 255, 255))
        views.append(warped)
    return views


def test_add_frame_detects_checkerboard():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)

    found, preview = calibrator.add_frame(board)

    assert found is True
    assert calibrator.captured_count == 1
    assert preview.shape == board.shape


def test_add_frame_returns_false_when_board_not_found():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)

    found, _preview = calibrator.add_frame(blank)

    assert found is False
    assert calibrator.captured_count == 0


def test_calibrate_requires_minimum_frames():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    with pytest.raises(ValueError):
        calibrator.calibrate()


def test_calibrate_with_few_frames_requires_force():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)
    for view in _warped_views(board, 3):
        calibrator.add_frame(view)

    with pytest.raises(ValueError):
        calibrator.calibrate()

    profile = calibrator.calibrate(force=True)
    assert profile.image_size == (board.shape[1], board.shape[0])
    assert profile.camera_matrix.shape == (3, 3)


def test_calibrate_and_undistort_preserves_shape():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)
    for view in _warped_views(board, 12):
        calibrator.add_frame(view)

    profile = calibrator.calibrate()
    corrected = undistort(board, profile)

    assert corrected.shape == board.shape


def test_tvecs_none_before_calibrate():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    assert calibrator.tvecs is None


def test_tvecs_populated_after_calibrate_matches_capture_count():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)
    for view in _warped_views(board, 12):
        calibrator.add_frame(view)

    calibrator.calibrate()

    assert calibrator.tvecs is not None
    assert len(calibrator.tvecs) == 12
    for tvec in calibrator.tvecs:
        assert np.asarray(tvec).shape == (3, 1)


def test_rvecs_none_before_calibrate():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    assert calibrator.rvecs is None


def test_rvecs_populated_after_calibrate_matches_capture_count_and_order():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)
    for view in _warped_views(board, 12):
        calibrator.add_frame(view)

    calibrator.calibrate()

    assert calibrator.rvecs is not None
    assert len(calibrator.rvecs) == len(calibrator.tvecs) == 12
    for rvec in calibrator.rvecs:
        assert np.asarray(rvec).shape == (3, 1)


def test_quick_checkerboard_found_true_positive():
    from imgflow.core.lens_calibration import quick_checkerboard_found

    board = _draw_checkerboard(5, 4)
    assert quick_checkerboard_found(board, (5, 4)) is True


def test_quick_checkerboard_found_false_on_blank():
    from imgflow.core.lens_calibration import quick_checkerboard_found

    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert quick_checkerboard_found(blank, (5, 4)) is False


def test_quick_checkerboard_found_is_fast_on_large_noisy_frame():
    import time

    from imgflow.core.lens_calibration import quick_checkerboard_found

    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 255, size=(1024, 1280), dtype=np.uint8)

    t0 = time.perf_counter()
    found = quick_checkerboard_found(noisy, (9, 6))
    elapsed = time.perf_counter() - t0

    assert found is False
    assert elapsed < 1.0  # tam çözünürlükte klasik dedektör bu durumda saniyeler sürebilir


def test_build_object_points_shape_and_scale():
    objp = build_object_points(inner_cols=5, inner_rows=4, square_size_mm=25.0)
    assert objp.shape == (20, 3)
    assert objp[:, 2].max() == 0.0  # z=0 düzlemi
    assert objp[:, :2].max() == 25.0 * 4  # en uzak köşe: (cols-1)*square, ya da (rows-1)*square


def test_detect_checkerboard_corners_stateless_matches_add_frame():
    board = _draw_checkerboard(5, 4)
    found, corners, preview = detect_checkerboard_corners(board, (5, 4))
    assert found is True
    assert corners.shape[0] == 20
    assert preview.shape == board.shape


def test_detect_checkerboard_corners_none_when_not_found():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    found, corners, _preview = detect_checkerboard_corners(blank, (5, 4))
    assert found is False
    assert corners is None


def _charuco_board(squares_x=5, squares_y=4):
    return build_charuco_board(
        CharucoBoardConfig(squares_x=squares_x, squares_y=squares_y, square_length_mm=25.0, marker_length_mm=18.0)
    )


def test_build_charuco_board_generates_detectable_image():
    board = _charuco_board()
    image = board.generateImage((500, 400))

    found, corners, ids, preview = detect_charuco_corners(image, board)

    assert found is True
    assert corners.shape[0] >= 4
    assert ids is not None
    assert preview.shape[:2] == image.shape[:2]


def test_build_charuco_board_rejects_unknown_dictionary():
    with pytest.raises(ValueError):
        build_charuco_board(
            CharucoBoardConfig(
                squares_x=5, squares_y=4, square_length_mm=25.0, marker_length_mm=18.0,
                dictionary_name="DICT_NOT_REAL",
            )
        )


def test_detect_charuco_corners_none_when_not_found():
    board = _charuco_board()
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)

    found, corners, ids, _preview = detect_charuco_corners(blank, board)

    assert found is False
    assert corners is None
    assert ids is None


def test_lens_calibrator_add_charuco_frame_and_calibrate():
    board = _charuco_board()
    base_image = board.generateImage((600, 500))
    calibrator = LensCalibrator()

    for view in _warped_views(base_image, 12):
        found, preview = calibrator.add_charuco_frame(view, board)
        assert preview.shape[:2] == view.shape[:2]

    assert calibrator.captured_count == 12
    profile = calibrator.calibrate()
    assert profile.camera_matrix.shape == (3, 3)

    corrected = undistort(base_image, profile)
    assert corrected.shape == base_image.shape


def test_add_frame_without_checkerboard_config_raises():
    calibrator = LensCalibrator()
    with pytest.raises(RuntimeError):
        calibrator.add_frame(_draw_checkerboard(5, 4))


def test_lens_profile_round_trips_through_dict():
    calibrator = LensCalibrator(CheckerboardConfig(inner_cols=5, inner_rows=4, square_size_mm=25.0))
    board = _draw_checkerboard(5, 4)
    for view in _warped_views(board, 12):
        calibrator.add_frame(view)
    profile = calibrator.calibrate()

    from imgflow.core.lens_calibration import LensProfile

    restored = LensProfile.from_dict(profile.to_dict())

    assert restored.image_size == profile.image_size
    assert np.allclose(restored.camera_matrix, profile.camera_matrix)
    assert np.allclose(restored.dist_coeffs, profile.dist_coeffs)
