import cv2
import numpy as np
import pytest

from imgflow.core.roi import RoiRect
from imgflow.core.shape_matching import (
    LabeledMatch,
    ShapeMatchingError,
    _nms,
    find_shape_model,
    render_match_overlay,
    train_shape_model,
)

_BASE_TRIANGLE = np.array([[0, -40], [35, 25], [-20, 30]], dtype=np.float64)
"""Asimetrik (skalen) üçgen — rotasyonel simetrisi yok, bu yüzden bulunan açı belirsizliksiz test edilebilir."""


def _draw_triangle(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (_BASE_TRIANGLE @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)


def _reference_image() -> np.ndarray:
    image = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(image, (100, 100), 0.0)
    return image


def _train():
    return train_shape_model(_reference_image(), RoiRect(50, 50, 100, 100))


def test_train_shape_model_extracts_points_per_level():
    model = _train()
    assert len(model.levels) == 3
    for level in model.levels:
        assert level.points.shape[0] > 0
        assert level.points.shape == (level.angles.shape[0], 2)


def test_train_shape_model_rejects_roi_without_edges():
    blank = np.full((200, 200), 255, dtype=np.uint8)
    with pytest.raises(ShapeMatchingError):
        train_shape_model(blank, RoiRect(50, 50, 100, 100))


def test_train_shape_model_rejects_tiny_roi():
    with pytest.raises(ShapeMatchingError):
        train_shape_model(_reference_image(), RoiRect(50, 50, 2, 2))


def test_find_shape_model_recovers_translation_and_rotation():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert len(matches) == 1
    match = matches[0]
    assert match.x == pytest.approx(130, abs=3)
    assert match.y == pytest.approx(80, abs=3)
    assert match.angle == pytest.approx(30, abs=3)
    assert match.score >= 0.5


def test_find_shape_model_exact_pose_scores_near_one():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (105, 95), 0.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert len(matches) == 1
    assert matches[0].score == pytest.approx(1.0, abs=0.05)


def test_find_shape_model_returns_empty_on_blank_image():
    model = _train()
    blank = np.full((200, 200), 255, dtype=np.uint8)

    matches = find_shape_model(blank, model, min_score=0.5, greediness=0.7, max_matches=1)

    assert matches == []


def test_find_shape_model_finds_multiple_instances():
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=2)

    assert len(matches) == 2
    positions = {(round(m.x / 10) * 10, round(m.y / 10) * 10) for m in matches}
    assert (80, 80) in positions
    assert (220, 200) in positions


def test_render_match_overlay_returns_bgr_image_same_size():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)
    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)
    entries = [LabeledMatch(label=f"ucgen{i}", model=model, match=m) for i, m in enumerate(matches, start=1)]

    overlay = render_match_overlay(search, entries)

    assert overlay.shape == (200, 200, 3)


def test_render_match_overlay_handles_no_matches():
    search = np.full((200, 200), 255, dtype=np.uint8)

    overlay = render_match_overlay(search, [])

    assert overlay.shape == (200, 200, 3)


def test_render_match_overlay_draws_label_text_for_each_entry():
    model = _train()
    search = np.full((200, 200), 255, dtype=np.uint8)
    _draw_triangle(search, (130, 80), 30.0)
    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=1)
    entries = [LabeledMatch(label="ucgen1", model=model, match=matches[0])]

    with_label = render_match_overlay(search, entries)
    without_label = render_match_overlay(search, [])

    # Etiket/kontur çizildiği için piksel farklılığı olmalı (aynı görüntü dönmemeli).
    assert not np.array_equal(with_label, without_label)


def test_find_shape_model_auto_mode_returns_all_matches_above_threshold():
    """min_score=0.7 (operatörün gerçek varsayılanı) kullanılıyor: otomatik modda (max_matches=
    None) TÜM eşiği geçen adaylar döner — düşük bir eşikle (ör. 0.5) bu üçgen model için yanlış
    açıdaki zayıf-ama-eşik-üstü sahte adaylar da (gerçek nesne değil) sonuca girebilir; bu,
    NMS'in bir hatası değil, 'otomatik'in tanımının doğal bir sonucudur (eşik altına düşen hiçbir
    şey elenmeden döner) — bu yüzden gerçekçi bir eşik kullanmak önemlidir."""
    model = _train()
    search = np.full((300, 300), 255, dtype=np.uint8)
    _draw_triangle(search, (80, 80), 10.0)
    _draw_triangle(search, (220, 200), -60.0)

    matches = find_shape_model(search, model, min_score=0.7, greediness=0.7, max_matches=None)

    assert len(matches) == 2


def test_find_shape_model_does_not_return_duplicate_detections_for_same_object():
    """Regresyon testi: NMS mesafe eşiği model boyutuna göre ölçeklenmeden önce, büyük/uzun bir
    model için aynı fiziksel nesnenin etrafında birden fazla yakın-konumlu aday birleştirilmeden
    ayrı 'eşleşme' olarak dönüyordu (kullanıcı raporu: "aynı görüntüyü birden fazla kez seçiyor")."""
    reference = np.full((300, 300), 255, dtype=np.uint8)
    _draw_elongated_shape(reference, (150, 150), 0.0)
    model = train_shape_model(reference, RoiRect(60, 110, 190, 80))

    search = np.full((400, 400), 255, dtype=np.uint8)
    _draw_elongated_shape(search, (150, 150), 20.0)
    _draw_elongated_shape(search, (280, 280), -50.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=10)

    assert len(matches) == 2


def test_nms_merges_same_position_candidates_despite_differing_angle():
    """Regresyon testi: eskiden NMS konum YETERLİ olsa bile açı farkı bir eşiği (1°) aşarsa
    adayları birleştirmiyordu — kaba/ince arama adımlarının ürettiği küçük açı sapmalarıyla
    AYNI fiziksel nesne için birden fazla 'eşleşme' dönüyordu (gerçek kullanıcı raporu: "aynı
    cismi farklı açılardan tespit edip tekrar tekrar gösteriyor"). Artık açı hiç dikkate
    alınmıyor — konum tek başına yeterli."""
    candidates = [
        [100.0, 100.0, 10.0, 0.95],
        [100.5, 99.5, 25.0, 0.90],  # aynı konum, 15° farklı açı -- eskiden AYRI kalırdı
    ]

    merged = _nms(candidates, dist_thresh=3.0)

    assert len(merged) == 1
    assert merged[0][3] == pytest.approx(0.95)  # en yüksek skorlu aday tutulur


def test_nms_keeps_distinct_positions_separate_regardless_of_angle():
    candidates = [[100.0, 100.0, 10.0, 0.95], [200.0, 100.0, 10.0, 0.90]]

    merged = _nms(candidates, dist_thresh=3.0)

    assert len(merged) == 2


def test_find_shape_model_keeps_close_but_distinct_objects_separate():
    """NMS eşiğinin model boyutuna oranlanması, gerçekten AYRI ama birbirine yakın iki nesneyi
    yanlışlıkla tek eşleşmeye indirmemeli."""
    reference = np.full((300, 300), 255, dtype=np.uint8)
    _draw_elongated_shape(reference, (150, 150), 0.0)
    model = train_shape_model(reference, RoiRect(60, 110, 190, 80))

    search = np.full((500, 500), 255, dtype=np.uint8)
    _draw_elongated_shape(search, (150, 150), 0.0)
    _draw_elongated_shape(search, (300, 150), 0.0)

    matches = find_shape_model(search, model, min_score=0.5, greediness=0.7, max_matches=10)

    assert len(matches) == 2


def _draw_elongated_shape(image: np.ndarray, center: tuple[float, float], angle_deg: float) -> None:
    base = np.array([[-60, -10], [60, -10], [60, 10], [-60, 10], [70, 0]], dtype=np.float64)
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]])
    pts = (base @ rot.T) + np.array(center)
    cv2.fillPoly(image, [pts.astype(np.int32)], color=0)
