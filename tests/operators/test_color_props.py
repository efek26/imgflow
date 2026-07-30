import numpy as np
import pytest

from imgflow.operators import registry
from imgflow.operators.builtin.color_props import ColorPropsOp


def test_registered():
    assert registry.get("analysis.color_props") is not None


def test_solid_gray_image_has_near_zero_ab():
    # Nötr gri (B=G=R eşit) LAB'da a*≈0, b*≈0 olmalı; L* orta parlaklıkta olmalı.
    image = np.full((30, 40, 3), 128, dtype=np.uint8)

    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    assert m["a_mean"] == pytest.approx(0.0, abs=2.0)
    assert m["b_mean"] == pytest.approx(0.0, abs=2.0)
    assert 40.0 < m["l_mean"] < 65.0
    assert out["overlay"].shape == (30, 40, 3)


def test_solid_red_image_has_positive_a():
    # BGR'de saf kırmızı -> LAB'da belirgin pozitif a* (kırmızı ekseni).
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[..., 2] = 255  # R kanalı

    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    assert m["a_mean"] > 30.0


def test_grayscale_image_supported():
    image = np.full((20, 20), 100, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    assert len(out["measurements"]) == 1


def test_tolerance_disabled_by_default_no_delta_e_key():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    assert "delta_e" not in out["measurements"][0]
    assert "tolerance_ok" not in out["measurements"][0]


def test_tolerance_enabled_matching_reference_is_ok():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    out = ColorPropsOp().run({"image": image}, {})
    m = out["measurements"][0]

    params = {
        "tolerance_enabled": True,
        "ref_l": m["l_mean"],
        "ref_a": m["a_mean"],
        "ref_b": m["b_mean"],
        "delta_e_max": 1.0,
    }
    out2 = ColorPropsOp().run({"image": image}, params)
    m2 = out2["measurements"][0]
    assert m2["tolerance_ok"] is True
    assert m2["delta_e"] == pytest.approx(0.0, abs=0.5)


def test_tolerance_enabled_far_reference_is_ng():
    image = np.full((10, 10, 3), 128, dtype=np.uint8)
    params = {
        "tolerance_enabled": True,
        "ref_l": 0.0,
        "ref_a": 100.0,
        "ref_b": 100.0,
        "delta_e_max": 5.0,
    }
    out = ColorPropsOp().run({"image": image}, params)
    m = out["measurements"][0]
    assert m["tolerance_ok"] is False


def _two_blob_image() -> np.ndarray:
    # İki kare de AYNI gri-seviye parlaklığa (~100) sahip ama farklı renkte -- Otsu
    # eşiklemesi (parlaklık bazlı) ikisini de tek bir "ön plan" sınıfına koyar, ayrım bağlı
    # bileşen analizinin aralarındaki boşluğu görmesiyle yapılır (bkz. core/auto_objects.py).
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 40, 255)  # kırmızıya çalan (BGR) -> belirgin pozitif a*
    image[25:35, 20:30] = (255, 121, 0)  # maviye çalan (BGR) -> farklı a*/b*
    return image


def test_per_object_disabled_by_default_still_single_measurement():
    out = ColorPropsOp().run({"image": _two_blob_image()}, {})
    assert len(out["measurements"]) == 1


def test_per_object_enabled_lists_each_blob_separately():
    out = ColorPropsOp().run({"image": _two_blob_image()}, {"per_object_enabled": True})
    measurements = out["measurements"]

    assert len(measurements) == 2
    for m in measurements:
        assert "label" in m
        assert {"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= m.keys()
    # Kırmızı kare belirgin pozitif a*, mavi kare belirgin farklı a* değeri taşımalı --
    # tek bir birleşik "tüm görüntü ortalaması" yerine gerçekten AYRI ölçüldüklerini doğrular.
    a_means = sorted(m["a_mean"] for m in measurements)
    assert a_means[1] - a_means[0] > 10.0
    assert out["overlay"].shape == _two_blob_image().shape


def test_per_object_enabled_applies_tolerance_per_object():
    params = {
        "per_object_enabled": True,
        "tolerance_enabled": True,
        "ref_l": 50.0,
        "ref_a": 0.0,
        "ref_b": 0.0,
        "delta_e_max": 5.0,
    }
    out = ColorPropsOp().run({"image": _two_blob_image()}, params)
    measurements = out["measurements"]

    assert len(measurements) == 2
    for m in measurements:
        assert "delta_e" in m
        assert "tolerance_ok" in m


def test_per_object_min_area_filters_small_blobs():
    image = _two_blob_image()
    image[0:2, 0:2] = (0, 255, 0)  # küçük gürültü lekesi

    out = ColorPropsOp().run({"image": image}, {"per_object_enabled": True, "min_object_area": 10.0})
    assert len(out["measurements"]) == 2


def test_per_object_max_area_filters_large_blobs():
    image = np.zeros((60, 60, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 40, 255)
    image[25:35, 20:30] = (255, 121, 0)
    image[40:60, 40:60] = (0, 40, 255)  # 400 px² -- diğerlerinden çok büyük

    out = ColorPropsOp().run(
        {"image": image}, {"per_object_enabled": True, "max_object_area": 150.0}
    )
    assert len(out["measurements"]) == 2


def test_per_object_manual_threshold_recovers_dim_blob():
    # Otsu'nun tek başına yakalayamadığı, parlaklığı diğerinden çok farklı ikinci bir nesne
    # (bkz. core/auto_objects.py testi) manuel eşik ile yakalanabilmeli.
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[5:15, 5:15] = (0, 0, 255)  # kırmızı, gray≈76
    image[25:35, 20:30] = (255, 0, 0)  # mavi, gray≈29

    otsu_out = ColorPropsOp().run({"image": image}, {"per_object_enabled": True})
    assert len(otsu_out["measurements"]) == 1

    manual_out = ColorPropsOp().run(
        {"image": image},
        {"per_object_enabled": True, "threshold_mode": "manual", "threshold_value": 15},
    )
    assert len(manual_out["measurements"]) == 2


def test_per_object_fill_holes_keeps_object_as_single_measurement():
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    image[5:25, 5:25] = (0, 40, 200)  # nesne gövdesi
    image[13:17, 13:17] = (0, 0, 0)  # ortasında eşik-altı bir "delik"

    out = ColorPropsOp().run(
        {"image": image}, {"per_object_enabled": True, "fill_holes": True}
    )
    assert len(out["measurements"]) == 1


def test_per_object_enabled_reports_min_max_range_per_channel():
    # Gerçek kullanıcı isteği: "lab analizi yaparken nesne tespitinde dalgalanma gözüksün" --
    # Elle ROI Çiz'deki min/max aralığı artık Otomatik Nesne Tespiti'nde de hesaplanmalı.
    # Tek bir bağlı bileşen (nesnenin TAMAMI arka plandan -- siyah -- belirgin şekilde daha
    # parlak) içinde yumuşak bir gri-seviye gradyanı: Otsu ikisini AYIRMAZ (tek nesne kalır),
    # ama nesnenin İÇİNDE gerçek bir parlaklık dalgalanması olur.
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    gradient = np.linspace(100, 220, 20).astype(np.uint8)
    for i, v in enumerate(gradient):
        image[5:25, 5 + i] = (int(v), int(v), int(v))

    out = ColorPropsOp().run({"image": image}, {"per_object_enabled": True})
    measurements = out["measurements"]

    assert len(measurements) == 1
    m = measurements[0]
    assert m["l_max"] - m["l_min"] > 30.0
    assert m["l_min"] < m["l_mean"] < m["l_max"]


def test_manual_roi_disabled_by_default_ignores_manual_rois():
    # Otomatik Nesne Tespiti/manuel ROI ikisi de kapalıyken tüm görüntü tek satır olmalı,
    # manual_rois dolu olsa bile.
    image = _two_blob_image()
    out = ColorPropsOp().run(
        {"image": image}, {"manual_rois": "[[0, 0, 10, 10]]"}
    )
    assert len(out["measurements"]) == 1


def test_manual_roi_enabled_ignores_auto_detection_uses_drawn_rects():
    image = _two_blob_image()
    params = {
        "manual_roi_enabled": True,
        "per_object_enabled": True,  # açık olsa bile manuel mod önceliklidir
        "manual_rois": "[[5, 5, 10, 10], [20, 25, 10, 10]]",
    }
    out = ColorPropsOp().run({"image": image}, params)
    measurements = out["measurements"]

    assert len(measurements) == 2
    assert [m["label"] for m in measurements] == [1, 2]
    assert all(m["manual"] is True for m in measurements)
    assert all({"bbox_x", "bbox_y", "bbox_w", "bbox_h"} <= m.keys() for m in measurements)
    # Kırmızı ROI belirgin pozitif a*, mavi ROI belirgin farklı a* değeri taşımalı.
    a_means = sorted(m["a_mean"] for m in measurements)
    assert a_means[1] - a_means[0] > 10.0


def test_manual_roi_reports_min_max_range_per_channel():
    # Yarısı koyu yarısı açık bir ROI -- ortalama tek başına bu dalgalanmayı gizler,
    # min/max aralığı bunu ortaya çıkarmalı (gerçek kullanıcı isteği).
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    image[:, :10] = (0, 0, 0)
    image[:, 10:] = (255, 255, 255)

    out = ColorPropsOp().run(
        {"image": image},
        {"manual_roi_enabled": True, "manual_rois": "[[0, 0, 20, 20]]"},
    )
    m = out["measurements"][0]

    assert m["l_max"] - m["l_min"] > 50.0
    assert m["l_min"] < m["l_mean"] < m["l_max"]


def test_manual_roi_applies_tolerance_per_roi():
    image = _two_blob_image()
    params = {
        "manual_roi_enabled": True,
        "manual_rois": "[[5, 5, 10, 10], [20, 25, 10, 10]]",
        "tolerance_enabled": True,
        "ref_l": 50.0,
        "ref_a": 0.0,
        "ref_b": 0.0,
        "delta_e_max": 5.0,
    }
    out = ColorPropsOp().run({"image": image}, params)
    measurements = out["measurements"]

    assert len(measurements) == 2
    for m in measurements:
        assert "delta_e" in m
        assert "tolerance_ok" in m


def test_manual_roi_empty_list_yields_no_measurements():
    image = _two_blob_image()
    out = ColorPropsOp().run(
        {"image": image}, {"manual_roi_enabled": True, "manual_rois": "[]"}
    )
    assert out["measurements"] == []
    assert out["overlay"].shape == image.shape
