import pytest

from imgflow.core.height_scale_calibration import HeightScaleModel, TeachPoint


def _synthetic_point(f_eff: float, h_camera: float, height_mm: float, real_mm: float = 20.0) -> TeachPoint:
    scale = f_eff / (h_camera - height_mm)
    return TeachPoint(height_mm=height_mm, pixel_distance=scale * real_mm, real_mm=real_mm)


def test_fit_recovers_known_parameters_from_synthetic_data():
    true_f, true_h = 1500.0, 500.0
    model = HeightScaleModel()
    for h in [50.0, 100.0, 150.0, 200.0]:
        model.add_point(_synthetic_point(true_f, true_h, h))

    model.fit()

    assert model.f_eff == pytest.approx(true_f, rel=1e-6)
    assert model.h_camera == pytest.approx(true_h, rel=1e-6)
    assert model.predict_scale(75.0) == pytest.approx(true_f / (true_h - 75.0), rel=1e-6)
    assert all(abs(r) < 1e-9 for r in model.residuals_mm_per_px())


def test_fit_requires_at_least_two_distinct_heights():
    model = HeightScaleModel()
    model.add_point(_synthetic_point(1500.0, 500.0, 100.0))
    model.add_point(_synthetic_point(1500.0, 500.0, 100.0))  # aynı yükseklik, farklı ölçüm

    with pytest.raises(ValueError):
        model.fit()


def test_fit_rejects_inconsistent_data_with_positive_slope():
    model = HeightScaleModel()
    # yükseklik arttıkça ölçek KÜÇÜLÜYORMUŞ gibi (fiziksel olarak tutarsız) veri üretelim
    model.add_point(TeachPoint(height_mm=50.0, pixel_distance=100.0, real_mm=20.0))
    model.add_point(TeachPoint(height_mm=100.0, pixel_distance=50.0, real_mm=20.0))

    with pytest.raises(ValueError):
        model.fit()


def test_predict_scale_before_fit_raises():
    model = HeightScaleModel()
    with pytest.raises(RuntimeError):
        model.predict_scale(100.0)


def test_predict_scale_rejects_height_at_or_above_camera_height():
    model = HeightScaleModel()
    for h in [50.0, 100.0]:
        model.add_point(_synthetic_point(1500.0, 500.0, h))
    model.fit()

    with pytest.raises(ValueError):
        model.predict_scale(500.0)
