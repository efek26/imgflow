import numpy as np
import pytest

from imgflow.core.focus_distance import FocusDistanceModel, FocusTeachPoint
from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint
from imgflow.core.lens_calibration import LensProfile
from imgflow.io_utils.calibration_store import (
    CalibrationProfileData,
    format_profile_label,
    list_profiles,
    load_profile,
    profile_mm_per_px,
    profile_name_from_label,
    save_profile,
)


def _sample_lens_profile(rms_error: float = 0.25) -> LensProfile:
    return LensProfile(
        camera_matrix=np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]]),
        dist_coeffs=np.array([[0.01, -0.02, 0.0, 0.0, 0.0]]),
        image_size=(640, 480),
        rms_error=rms_error,
    )


def _sample_height_model() -> HeightScaleModel:
    model = HeightScaleModel()
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=15.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=20.0, real_mm=20.0))
    model.fit()
    return model


def test_save_and_load_profile_round_trips_both_calibrations(tmp_path):
    lens_profile = _sample_lens_profile()
    height_model = _sample_height_model()

    save_profile("hat1", lens_profile, height_model, directory=tmp_path)
    data = load_profile("hat1", directory=tmp_path)

    assert data.lens_profile is not None
    assert np.allclose(data.lens_profile.camera_matrix, lens_profile.camera_matrix)
    assert data.lens_profile.image_size == lens_profile.image_size

    assert data.height_model is not None
    assert data.height_model.f_eff == height_model.f_eff
    assert data.height_model.h_camera == height_model.h_camera
    assert len(data.height_model.points) == 2


def test_save_and_load_profile_with_only_height_model(tmp_path):
    height_model = _sample_height_model()
    save_profile("hat2", height_model=height_model, directory=tmp_path)

    data = load_profile("hat2", directory=tmp_path)

    assert data.lens_profile is None
    assert data.height_model is not None


def _sample_focus_model() -> FocusDistanceModel:
    model = FocusDistanceModel()
    model.add_point(FocusTeachPoint(focus_value=10.0, distance_mm=100.0))
    model.add_point(FocusTeachPoint(focus_value=20.0, distance_mm=150.0))
    model.fit()
    return model


def test_save_and_load_profile_round_trips_focus_model(tmp_path):
    focus_model = _sample_focus_model()
    save_profile("hat_focus", focus_model=focus_model, directory=tmp_path)

    data = load_profile("hat_focus", directory=tmp_path)

    assert data.focus_model is not None
    assert data.focus_model.slope == focus_model.slope
    assert data.focus_model.intercept == focus_model.intercept
    assert data.focus_model.correlation == focus_model.correlation
    assert len(data.focus_model.points) == 2


def test_load_profile_without_focus_model_defaults_to_none(tmp_path):
    save_profile("hat_no_focus", lens_profile=_sample_lens_profile(), directory=tmp_path)

    data = load_profile("hat_no_focus", directory=tmp_path)

    assert data.focus_model is None


def test_saving_only_focus_model_preserves_previously_saved_lens_profile(tmp_path):
    lens_profile = _sample_lens_profile()
    save_profile("hat_combo", lens_profile=lens_profile, directory=tmp_path)

    focus_model = _sample_focus_model()
    save_profile("hat_combo", focus_model=focus_model, directory=tmp_path)  # lens_profile verilmedi

    data = load_profile("hat_combo", directory=tmp_path)
    assert data.focus_model is not None
    assert data.lens_profile is not None  # önceki kaydedilen lens profili KAYBOLMAMALI
    assert np.allclose(data.lens_profile.camera_matrix, lens_profile.camera_matrix)


def test_save_and_load_profile_round_trips_reference_distance(tmp_path):
    save_profile("hat_ref", reference_distance_mm=830.5, directory=tmp_path)

    data = load_profile("hat_ref", directory=tmp_path)

    assert data.reference_distance_mm == pytest.approx(830.5)


def test_load_profile_without_reference_distance_defaults_to_none(tmp_path):
    save_profile("hat_no_ref", lens_profile=_sample_lens_profile(), directory=tmp_path)

    data = load_profile("hat_no_ref", directory=tmp_path)

    assert data.reference_distance_mm is None


def test_saving_only_reference_distance_preserves_previously_saved_lens_profile(tmp_path):
    lens_profile = _sample_lens_profile()
    save_profile("hat_ref_combo", lens_profile=lens_profile, directory=tmp_path)

    save_profile("hat_ref_combo", reference_distance_mm=750.0, directory=tmp_path)  # lens_profile verilmedi

    data = load_profile("hat_ref_combo", directory=tmp_path)
    assert data.reference_distance_mm == pytest.approx(750.0)
    assert data.lens_profile is not None  # önceki kaydedilen lens profili KAYBOLMAMALI
    assert np.allclose(data.lens_profile.camera_matrix, lens_profile.camera_matrix)


def test_resaving_lens_profile_preserves_previously_saved_reference_distance(tmp_path):
    save_profile("hat_ref_combo2", reference_distance_mm=900.0, directory=tmp_path)

    save_profile("hat_ref_combo2", lens_profile=_sample_lens_profile(), directory=tmp_path)  # reference verilmedi

    data = load_profile("hat_ref_combo2", directory=tmp_path)
    assert data.reference_distance_mm == pytest.approx(900.0)


def _sample_plane_rectification():
    from imgflow.core.plane_rectification import compute_plane_rectification

    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.1], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [800.0]])
    return compute_plane_rectification(K, rvec, tvec, (640, 480))


def test_save_and_load_profile_round_trips_plane_rectification(tmp_path):
    rectification = _sample_plane_rectification()
    save_profile("hat_rect", plane_rectification=rectification, directory=tmp_path)

    data = load_profile("hat_rect", directory=tmp_path)

    assert data.plane_rectification is not None
    assert np.allclose(data.plane_rectification.homography, rectification.homography)
    assert data.plane_rectification.output_size == rectification.output_size
    assert data.plane_rectification.mm_per_px == pytest.approx(rectification.mm_per_px)


def test_load_profile_without_plane_rectification_defaults_to_none(tmp_path):
    save_profile("hat_no_rect", lens_profile=_sample_lens_profile(), directory=tmp_path)

    data = load_profile("hat_no_rect", directory=tmp_path)

    assert data.plane_rectification is None


def test_saving_only_plane_rectification_preserves_previously_saved_lens_profile(tmp_path):
    lens_profile = _sample_lens_profile()
    save_profile("hat_rect_combo", lens_profile=lens_profile, directory=tmp_path)

    save_profile("hat_rect_combo", plane_rectification=_sample_plane_rectification(), directory=tmp_path)

    data = load_profile("hat_rect_combo", directory=tmp_path)
    assert data.plane_rectification is not None
    assert data.lens_profile is not None  # önceki kaydedilen lens profili KAYBOLMAMALI
    assert np.allclose(data.lens_profile.camera_matrix, lens_profile.camera_matrix)


def test_list_profiles_returns_sorted_names(tmp_path):
    save_profile("b_profile", directory=tmp_path)
    save_profile("a_profile", directory=tmp_path)

    assert list_profiles(tmp_path) == ["a_profile", "b_profile"]


def test_list_profiles_empty_when_directory_missing(tmp_path):
    assert list_profiles(tmp_path / "does_not_exist") == []


def test_save_defaults_created_at_when_not_given(tmp_path):
    save_profile("hat3", lens_profile=_sample_lens_profile(), directory=tmp_path)

    data = load_profile("hat3", directory=tmp_path)

    assert data.created_at is not None
    assert data.operator_note is None


def test_save_accepts_explicit_created_at_and_operator_note(tmp_path):
    save_profile(
        "hat4",
        lens_profile=_sample_lens_profile(),
        directory=tmp_path,
        created_at="2026-01-15T10:00:00",
        operator_note="Efe - ilk kurulum",
    )

    data = load_profile("hat4", directory=tmp_path)

    assert data.created_at == "2026-01-15T10:00:00"
    assert data.operator_note == "Efe - ilk kurulum"


def test_saving_only_lens_profile_preserves_previously_saved_height_model(tmp_path):
    height_model = _sample_height_model()
    save_profile("hat5", height_model=height_model, directory=tmp_path)

    lens_profile = _sample_lens_profile()
    save_profile("hat5", lens_profile=lens_profile, directory=tmp_path)  # height_model verilmedi

    data = load_profile("hat5", directory=tmp_path)
    assert data.lens_profile is not None
    assert data.height_model is not None  # önceki kaydedilen model KAYBOLMAMALI
    assert data.height_model.f_eff == height_model.f_eff


def test_saving_only_height_model_preserves_previously_saved_lens_profile(tmp_path):
    lens_profile = _sample_lens_profile()
    save_profile("hat6", lens_profile=lens_profile, directory=tmp_path)

    height_model = _sample_height_model()
    save_profile("hat6", height_model=height_model, directory=tmp_path)  # lens_profile verilmedi

    data = load_profile("hat6", directory=tmp_path)
    assert data.height_model is not None
    assert data.lens_profile is not None  # önceki kaydedilen lens profili KAYBOLMAMALI
    assert np.allclose(data.lens_profile.camera_matrix, lens_profile.camera_matrix)


def test_resaving_lens_profile_overwrites_previous_lens_profile(tmp_path):
    save_profile("hat7", lens_profile=_sample_lens_profile(rms_error=1.0), directory=tmp_path)
    save_profile("hat7", lens_profile=_sample_lens_profile(rms_error=0.1), directory=tmp_path)

    data = load_profile("hat7", directory=tmp_path)
    assert data.lens_profile.rms_error == 0.1


def _sample_plane_rectification(mm_per_px: float = 0.4):
    from imgflow.core.plane_rectification import compute_plane_rectification

    K = np.array([[1000.0, 0.0, 320.0], [0.0, 1000.0, 240.0], [0.0, 0.0, 1.0]])
    rvec = np.array([[0.1], [0.0], [0.0]])
    tvec = np.array([[0.0], [0.0], [800.0]])
    return compute_plane_rectification(K, rvec, tvec, (640, 480), mm_per_px=mm_per_px)


def test_profile_mm_per_px_prefers_plane_rectification():
    data = CalibrationProfileData(
        lens_profile=_sample_lens_profile(),
        height_model=None,
        created_at=None,
        operator_note=None,
        reference_distance_mm=830.0,  # 830/1000 = 0.83 -- rectification öncelikli olmalı
        plane_rectification=_sample_plane_rectification(mm_per_px=0.4),
    )

    assert profile_mm_per_px(data) == pytest.approx(0.4)


def test_profile_mm_per_px_falls_back_to_reference_distance_over_fx():
    data = CalibrationProfileData(
        lens_profile=_sample_lens_profile(),  # fx = 1000.0
        height_model=None,
        created_at=None,
        operator_note=None,
        reference_distance_mm=500.0,
    )

    assert profile_mm_per_px(data) == pytest.approx(0.5)


def test_profile_mm_per_px_is_none_without_a_static_scale_source():
    data = CalibrationProfileData(
        lens_profile=_sample_lens_profile(), height_model=_sample_height_model(), created_at=None, operator_note=None
    )

    assert profile_mm_per_px(data) is None


def test_format_profile_label_appends_mm_per_px_when_available():
    data = CalibrationProfileData(
        lens_profile=_sample_lens_profile(),
        height_model=None,
        created_at=None,
        operator_note=None,
        reference_distance_mm=420.0,
    )

    assert format_profile_label("hat1", data) == "hat1 (0.420 mm/px)"


def test_format_profile_label_returns_plain_name_without_a_static_scale_source():
    data = CalibrationProfileData(
        lens_profile=None, height_model=_sample_height_model(), created_at=None, operator_note=None
    )

    assert format_profile_label("hat2", data) == "hat2"


def test_profile_name_from_label_strips_mm_per_px_suffix():
    assert profile_name_from_label("hat1 (0.420 mm/px)") == "hat1"


def test_profile_name_from_label_returns_plain_name_unchanged():
    assert profile_name_from_label("hat2") == "hat2"
