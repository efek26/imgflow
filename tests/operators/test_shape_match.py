from dataclasses import replace

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


def _draw_shape(
    image: np.ndarray, base: np.ndarray, center: tuple[float, float], angle_deg: float, scale: float = 1.0
) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (base * scale @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _draw_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float, scale: float = 1.0) -> None:
    _draw_shape(image, _BASE_TRIANGLE, center, angle_deg, scale)


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
    # `displacement_*` her zaman mevcut (taze eğitilmiş model `reference_center` taşıyor,
    # bkz. `train_shape_model`) -- `mm_per_px` verilmediği için cm alanları YOK.
    assert set(match.keys()) == {
        "model", "label", "x", "y", "angle", "score",
        "width_px", "height_px",
        "displacement_px", "displacement_x_px", "displacement_y_px",
    }
    assert match["model"] == "ucgen"
    assert match["label"] == "1"
    assert match["x"] == pytest.approx(130, abs=3)
    assert match["y"] == pytest.approx(80, abs=3)
    assert match["angle"] == pytest.approx(30, abs=3)

    assert isinstance(out["overlay"], np.ndarray)
    assert out["overlay"].shape == (200, 200, 3)


def test_run_adds_size_and_displacement_when_mm_per_px_calibrated(saved_model_name):
    """Gerçek kullanıcı isteği: "geometrik eşlemede scale boyutu ve cismin ötelenme uzaklığı
    da yazmalı", sonraki turda netleşti: "ötelemeyi x,y şeklinde yaz ve kalibrasyon varsa cm
    şeklinde yaz" -- tek bir skaler mesafe yön bilgisini gizliyordu, artık işaretli x/y
    bileşenleri de cm cinsinden ayrı ayrı raporlanıyor."""
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {
            "model_names": saved_model_name,
            "min_score": 0.5,
            "greediness": 0.7,
            "auto_count": False,
            "num_matches": 1,
            "mm_per_px": 0.5,
        },
    )

    match = out["measurements"][0]
    # Eğitim ROI'si RoiRect(50,50,100,100) -> width/height_px SABİT 100 (ölçek araması yok).
    # Kalibrasyon aktifken bile px değeri HÂLÂ mevcut olmalı -- gerçek kullanıcı isteği:
    # "kalibrasyon seçili olduğu her senaryoda pixelin yanında mm de yazsın veya cm".
    assert match["width_px"] == pytest.approx(100.0)
    assert match["height_px"] == pytest.approx(100.0)
    assert match["width_mm"] == pytest.approx(50.0)
    assert match["height_mm"] == pytest.approx(50.0)
    reference_center = (100.0, 100.0)  # roi merkezi (50+100/2, 50+100/2)
    expected_dx = match["x"] - reference_center[0]
    expected_dy = match["y"] - reference_center[1]
    expected_displacement_px = (expected_dx**2 + expected_dy**2) ** 0.5
    assert match["displacement_px"] == pytest.approx(expected_displacement_px)
    assert match["displacement_x_px"] == pytest.approx(expected_dx)
    assert match["displacement_y_px"] == pytest.approx(expected_dy)
    assert match["displacement_cm"] == pytest.approx(expected_displacement_px * 0.5 / 10.0)
    assert match["displacement_x_cm"] == pytest.approx(expected_dx * 0.5 / 10.0)
    assert match["displacement_y_cm"] == pytest.approx(expected_dy * 0.5 / 10.0)


def test_run_without_mm_per_px_has_no_cm_fields(saved_model_name):
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": saved_model_name, "min_score": 0.5, "greediness": 0.7, "auto_count": False, "num_matches": 1},
    )

    match = out["measurements"][0]
    assert "width_mm" not in match
    assert "height_mm" not in match
    assert "displacement_cm" not in match
    assert "displacement_x_cm" not in match
    assert "displacement_y_cm" not in match


def test_run_without_scale_search_has_no_scale_percent_field(saved_model_name):
    """Varsayılan `scale_min=scale_max=1.0` (Ölçek Araması KAPALI) iken `scale_percent` alanı
    HİÇ eklenmemeli -- eskisiyle BİREBİR aynı çıktı (bkz. `test_run_produces_measurements_
    and_overlay`'in tam alan seti karşılaştırması)."""
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": saved_model_name, "min_score": 0.5, "greediness": 0.7, "auto_count": False, "num_matches": 1},
    )

    assert "scale_percent" not in out["measurements"][0]


def test_run_with_scale_search_finds_larger_object_and_reports_scale_percent(saved_model_name):
    """Gerçek kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale boyutu yazmalı" --
    `scale_max > scale_min` verilince öğretilenden %50 büyük bir nesne bulunmalı VE
    `scale_percent` alanı gerçek ölçeğe (~150) yakın olmalı; `mm_per_px` de verilirse
    width/height_mm bulunan ölçeğe göre ayarlanmalı (SABİT öğretilen boyut DEĞİL)."""
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (150.0, 150.0), 0.0, scale=1.5)

    out = ShapeMatchOp().run(
        {"image": search},
        {
            "model_names": saved_model_name,
            "min_score": 0.5,
            "greediness": 0.7,
            "auto_count": False,
            "num_matches": 1,
            "scale_min": 0.8,
            "scale_max": 1.8,
            "scale_step_coarse": 0.2,
            "mm_per_px": 0.5,
        },
    )

    assert len(out["measurements"]) == 1
    match = out["measurements"][0]
    assert match["scale_percent"] == pytest.approx(150.0, abs=20.0)
    # Eğitim ROI'si RoiRect(50,50,100,100) -> öğretilen width/height_px = 100 (ölçek=1.0'da).
    # Bulunan ölçek ~1.5 olduğundan mm boyutu da SABİT 50mm değil, ona göre büyümeli.
    assert match["width_mm"] > 50.0 * 1.2


def test_run_with_old_model_missing_reference_center_has_no_displacement_field(tmp_path, monkeypatch):
    """Bu özellikten ÖNCE kaydedilmiş bir modelde `reference_center` yok (`None`) --
    `displacement_*` alanları HİÇ eklenmemeli (yanlış bir "0mm öteleme" izlenimi vermemek
    için), çökme de olmamalı; `mm_per_px` yine de width/height_mm'i etkiler."""
    monkeypatch.setattr(shape_model_store, "SHAPE_MODEL_DIR", tmp_path / "shape_models")
    reference = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(reference, (100, 100), 0.0)
    model = train_shape_model(reference, RoiRect(50, 50, 100, 100))
    old_style_model = replace(model, reference_center=None)
    shape_model_store.save_shape_model("eski_model", old_style_model)

    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    out = ShapeMatchOp().run(
        {"image": search},
        {
            "model_names": "eski_model",
            "min_score": 0.5,
            "greediness": 0.7,
            "auto_count": False,
            "num_matches": 1,
            "mm_per_px": 0.5,
        },
    )

    match = out["measurements"][0]
    assert "displacement_px" not in match
    assert "displacement_x_px" not in match
    assert "displacement_cm" not in match
    assert match["width_mm"] == pytest.approx(50.0)


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


def test_angle_step_coarse_param_is_forwarded_to_find_shape_model(saved_model_name, monkeypatch):
    from imgflow.operators.builtin import shape_match as shape_match_module

    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)
    received_kwargs = {}
    original = shape_match_module.find_shape_model

    def spy(*args, **kwargs):
        received_kwargs.update(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(shape_match_module, "find_shape_model", spy)

    ShapeMatchOp().run(
        {"image": search},
        {"model_names": saved_model_name, "angle_step_coarse": 7.5, "min_score": 0.5, "greediness": 0.7},
    )

    assert received_kwargs["angle_step_coarse"] == 7.5


def test_multiple_models_share_single_precomputed_search_pyramid(two_saved_models, monkeypatch):
    # Gerçek kullanıcı raporu: "şekil bulma çok kasıyor" -- birden fazla model seçiliyken
    # arama görüntüsünün gradyan piramidi artık HER model için değil, TEK SEFERDE kurulup
    # paylaşılıyor (bkz. `core/shape_matching.py::build_search_pyramid`).
    from imgflow.operators.builtin import shape_match as shape_match_module

    name_a, name_b = two_saved_models
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 0.0)
    _draw_rect(search, (220, 200), 0.0)
    build_calls = []
    original = shape_match_module.build_search_pyramid

    def spy(*args, **kwargs):
        build_calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(shape_match_module, "build_search_pyramid", spy)

    out = ShapeMatchOp().run(
        {"image": search},
        {"model_names": f"{name_a}, {name_b}", "min_score": 0.7, "greediness": 0.7, "auto_count": True},
    )

    assert len(build_calls) == 1  # iki model için değil, TEK SEFER kuruldu
    assert {m["model"] for m in out["measurements"]} == {name_a, name_b}


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
