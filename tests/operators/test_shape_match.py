import cv2
import numpy as np
import pytest

from imgflow.core.roi import RoiRect
from imgflow.core.shape_matching import train_shape_model
from imgflow.io_utils import shape_model_store
from imgflow.operators import registry
from imgflow.operators.builtin.shape_match import ShapeMatchOp

_BASE_TRIANGLE = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)
_BASE_RECT = np.array([[-35, -15], [35, -15], [35, 15], [-35, 15], [45, 0]], dtype=np.float64)


def _draw_shape(image: np.ndarray, base: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (base @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _draw_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    _draw_shape(image, _BASE_TRIANGLE, center, angle_deg)


def _draw_rect(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    _draw_shape(image, _BASE_RECT, center, angle_deg)


@pytest.fixture
def saved_model_name(tmp_path, monkeypatch):
    """Gerçek ~/.imgflow/shape_models dizinine ASLA dokunma — izole tmp_path'e yönlendir."""
    monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path / "shape_models")
    reference = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(reference, (100, 100), 0.0)
    model = train_shape_model(reference, RoiRect(50, 50, 100, 100))
    shape_model_store.save_shape_model("ucgen", model)
    return "ucgen"


@pytest.fixture
def two_saved_models(tmp_path, monkeypatch):
    """İki farklı isimde model kaydeder ('parca_a': üçgen, 'parca_b': dikdörtgen) — aynı adımda
    birden fazla model arama senaryolarını test etmek için."""
    monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path / "shape_models")

    ref_a = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(ref_a, (100, 100), 0.0)
    shape_model_store.save_shape_model("parca_a", train_shape_model(ref_a, RoiRect(50, 50, 100, 100)))

    ref_b = np.full((200, 200), 255, dtype=np.uint8)
    _draw_rect(ref_b, (100, 100), 0.0)
    shape_model_store.save_shape_model("parca_b", train_shape_model(ref_b, RoiRect(45, 65, 110, 70)))

    return "parca_a", "parca_b"


def test_registered():
    assert registry.get("geom.shape_match") is not None


def test_empty_model_names_raises_value_error():
    image = np.full((200, 200), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="model_names"):
        ShapeMatchOp().run({"image": image}, {"model_names": ""})


def test_unknown_model_name_raises_not_found(saved_model_name):
    image = np.full((200, 200), 255, dtype=np.uint8)
    with pytest.raises(shape_model_store.ShapeModelNotFoundError):
        ShapeMatchOp().run({"image": image}, {"model_names": "yok_boyle_model"})


def test_run_produces_measurements_and_overlay(saved_model_name):
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": saved_model_name, "min_score": 0.5, "greediness": 0.7, "auto_count": False, "num_matches": 1},
    )

    assert len(out["measurements"]) == 1
    match = out["measurements"][0]
    assert set(match.keys()) == {"model", "label", "x", "y", "angle", "score"}
    assert match["model"] == "ucgen"
    assert match["label"] == "1"
    assert match["x"] == pytest.approx(130, abs=3)
    assert match["y"] == pytest.approx(80, abs=3)
    assert match["angle"] == pytest.approx(30, abs=3)

    assert isinstance(out["overlay"], np.ndarray)
    assert out["overlay"].shape == (200, 200, 3)


def test_auto_count_finds_all_instances_above_threshold(saved_model_name):
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": saved_model_name, "min_score": 0.7, "greediness": 0.7, "auto_count": True},
    )

    assert len(out["measurements"]) == 2
    # Etiketler artık model adı+sırası değil, TÜM modeller genelinde tek bir akan sayaç.
    assert {m["label"] for m in out["measurements"]} == {"1", "2"}


def test_manual_count_limits_matches_when_auto_count_disabled(saved_model_name):
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {
            "model_names": saved_model_name,
            "min_score": 0.5,
            "greediness": 0.7,
            "auto_count": False,
            "num_matches": 1,
        },
    )

    assert len(out["measurements"]) == 1


def test_multiple_models_searched_in_one_run_and_labeled_sequentially(two_saved_models):
    """Etiketler artık model adı+sırası (ör. 'parca_a1') DEĞİL, TÜM modeller genelinde tek bir
    akan sayaç ('1', '2', ...) — hangi model olduğu ayrı 'model' alanında hâlâ mevcut."""
    name_a, name_b = two_saved_models
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 0.0)
    _draw_rect(search, (220, 200), 0.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": f"{name_a}, {name_b}", "min_score": 0.7, "greediness": 0.7, "auto_count": True},
    )

    models_found = {m["model"] for m in out["measurements"]}
    labels_found = {m["label"] for m in out["measurements"]}
    assert models_found == {name_a, name_b}
    assert labels_found == {"1", "2"}
    assert out["overlay"].shape == (300, 300, 3)
