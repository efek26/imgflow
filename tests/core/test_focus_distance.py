import numpy as np
import pytest

from imgflow.core.focus_distance import (
    FocusDistanceModel,
    FocusTeachPoint,
    model_from_calibration_frames,
)


def test_fit_requires_minimum_points():
    model = FocusDistanceModel()
    model.add_point(FocusTeachPoint(focus_value=10.0, distance_mm=100.0))

    with pytest.raises(ValueError):
        model.fit()


def test_fit_rejects_identical_focus_values():
    model = FocusDistanceModel()
    model.add_point(FocusTeachPoint(focus_value=5.0, distance_mm=100.0))
    model.add_point(FocusTeachPoint(focus_value=5.0, distance_mm=200.0))

    with pytest.raises(ValueError):
        model.fit()


def test_fit_recovers_linear_relationship_and_strong_correlation():
    model = FocusDistanceModel()
    for focus_value, distance in [(10.0, 100.0), (20.0, 150.0), (30.0, 200.0), (40.0, 250.0)]:
        model.add_point(FocusTeachPoint(focus_value=focus_value, distance_mm=distance))

    model.fit()

    assert model.slope == pytest.approx(5.0)
    assert model.intercept == pytest.approx(50.0)
    assert model.correlation == pytest.approx(1.0, abs=1e-6)


def test_predict_distance_clamps_to_calibrated_range():
    model = FocusDistanceModel()
    for focus_value, distance in [(10.0, 100.0), (20.0, 150.0), (30.0, 200.0)]:
        model.add_point(FocusTeachPoint(focus_value=focus_value, distance_mm=distance))
    model.fit()

    assert model.predict_distance(1000.0) == pytest.approx(200.0)
    assert model.predict_distance(-1000.0) == pytest.approx(100.0)
    assert model.predict_distance(20.0) == pytest.approx(150.0)


def test_predict_distance_before_fit_raises():
    model = FocusDistanceModel()
    with pytest.raises(RuntimeError):
        model.predict_distance(10.0)


def test_model_round_trips_through_dict():
    model = FocusDistanceModel()
    for focus_value, distance in [(10.0, 100.0), (20.0, 150.0), (30.0, 200.0)]:
        model.add_point(FocusTeachPoint(focus_value=focus_value, distance_mm=distance))
    model.fit()

    restored = FocusDistanceModel.from_dict(model.to_dict())

    assert restored.slope == pytest.approx(model.slope)
    assert restored.intercept == pytest.approx(model.intercept)
    assert restored.correlation == pytest.approx(model.correlation)
    assert len(restored.points) == len(model.points)


def test_model_from_calibration_frames_pairs_focus_with_tvec_distance():
    sharp = np.zeros((50, 50), dtype=np.uint8)
    sharp[::2, :] = 255  # yüksek frekanslı desen -> yüksek netlik ölçüsü
    blank = np.full((50, 50), 128, dtype=np.uint8)  # düz -> netlik ölçüsü 0

    frames = [sharp, blank]
    tvecs = [np.array([0.0, 0.0, 100.0]), np.array([0.0, 0.0, 300.0])]

    model = model_from_calibration_frames(frames, tvecs)

    assert len(model.points) == 2
    assert model.points[0].distance_mm == pytest.approx(100.0)
    assert model.points[1].distance_mm == pytest.approx(300.0)
    assert model.points[0].focus_value > model.points[1].focus_value
