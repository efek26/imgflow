import cv2
import numpy as np

from imgflow.operators import registry
from imgflow.operators.builtin.connected_components import ConnectedComponentsOp
from imgflow.operators.builtin.region_props import RegionPropsOp, draw_measurements_overlay


def test_registered():
    assert registry.get("analysis.region_props") is not None


def _labels_for(img: np.ndarray) -> np.ndarray:
    return ConnectedComponentsOp().run({"image": img}, {"connectivity": "8"})["labels"]


def test_measures_square_region():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255  # 10x10 kare, alan=100

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    assert len(out["measurements"]) == 1
    m = out["measurements"][0]
    assert m["area"] == 100.0
    assert m["bbox_w"] == 10
    assert m["bbox_h"] == 10
    assert abs(m["centroid_x"] - 9.5) < 0.5
    assert abs(m["centroid_y"] - 9.5) < 0.5
    assert m["perimeter"] > 0
    assert 0 < m["circularity"] <= 1.2


def test_min_area_filters_small_noise():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255
    img[1, 1] = 255  # 1 piksel gürültü

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 5})

    assert len(out["measurements"]) == 1


def test_min_area_is_interpreted_as_cm2_when_calibrated():
    """`tol_short_min`/`tol_long_min` ile AYNI aile: `min_area` kalibrasyonluyken (mm_per_px>0)
    cm², değilse ham px² olarak yorumlanır — gerçek kullanıcı isteği: 'min alanda
    kalibrasyonumuz çalışmalı'."""
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255  # 10x10 -> alan=100 px
    img[1, 1] = 255  # 1 piksel gürültü

    # mm_per_px=1.0 -> 100 pikselik bölge 1.0 cm², gürültü (1px) 0.01 cm².
    out = RegionPropsOp().run(
        {"labels": _labels_for(img)}, {"min_area": 0.5, "mm_per_px": 1.0}
    )
    assert len(out["measurements"]) == 1

    # Kalibrasyon yokken (mm_per_px=0) AYNI sayı ham px² olarak yorumlanır — 0.5 px² eşiği
    # gürültüyü (1px) elemez, her iki bölge de kalır.
    out_uncalibrated = RegionPropsOp().run(
        {"labels": _labels_for(img)}, {"min_area": 0.5, "mm_per_px": 0.0}
    )
    assert len(out_uncalibrated["measurements"]) == 2


def test_no_regions_returns_empty_list():
    img = np.zeros((20, 20), dtype=np.uint8)
    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})
    assert out["measurements"] == []


def test_overlay_is_produced_and_shows_region():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    overlay = out["overlay"]
    assert overlay.shape == (20, 20, 3)
    # Bölge içi (kutu çizgisinden ve köşelerinden uzak, merkeze yakın bir nokta) beyaz kalmalı.
    # Not: minAreaRect tabanlı çizim eski axis-aligned cv2.rectangle ile aynı değil; köşelere
    # yakın pikseller artık çizginin kalınlığına girebiliyor, bu yüzden köşe yerine iç merkez
    # kontrol edilir.
    assert overlay[12, 9].tolist() == [255, 255, 255]
    assert (overlay[5, 5:15] == [0, 255, 0]).all(axis=-1).any()  # yeşil sınırlayıcı kutu çizgisi


def test_overlay_with_no_regions_is_still_produced():
    img = np.zeros((20, 20), dtype=np.uint8)
    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})
    assert out["overlay"].shape == (20, 20, 3)


def test_draw_measurements_overlay_scales_lines_thicker_on_larger_images():
    """Gerçek kullanıcı raporu: sabit (küçük test görüntüsüne göre ayarlanmış) fontScale/
    çizgi kalınlığı, gerçek endüstriyel kamera çözünürlüğünde yazıları/kutuları orantısız
    derecede küçük gösteriyordu. Aynı boyuttaki bir kutu, daha büyük bir taban görüntüde daha
    KALIN çizilmeli (bkz. `region_props._TEXT_SCALE_REFERENCE_DIM`)."""
    measurement = {
        "bbox_x": 40,
        "bbox_y": 40,
        "bbox_w": 100,
        "bbox_h": 100,
        "obb_cx": 90.0,
        "obb_cy": 90.0,
        "obb_w": 100.0,
        "obb_h": 100.0,
        "obb_angle": 0.0,
    }
    small = np.zeros((200, 200, 3), dtype=np.uint8)
    small_overlay = draw_measurements_overlay(small, [measurement], 0.0)
    small_green = int((small_overlay == [0, 255, 0]).all(axis=-1).sum())

    large = np.zeros((3000, 3000, 3), dtype=np.uint8)
    large_overlay = draw_measurements_overlay(large, [measurement], 0.0)
    large_green = int((large_overlay == [0, 255, 0]).all(axis=-1).sum())

    assert large_green > small_green * 1.5


def test_draw_measurements_overlay_shows_cm_not_mm_when_calibrated():
    measurement = {
        "bbox_x": 5,
        "bbox_y": 5,
        "bbox_w": 10,
        "bbox_h": 10,
        "obb_cx": 10.0,
        "obb_cy": 10.0,
        "obb_w": 10.0,
        "obb_h": 10.0,
        "obb_angle": 0.0,
    }
    # Genişlik 220px -- "10 x 10 px (1.00 x 1.00 cm)" metninin (px+cm ikisi de yazıldığından,
    # bkz. `draw_measurements_overlay` docstring'i) TAMAMI görünür alana sığmalı; dar bir
    # tuval (ör. eski 60x60) metnin farklılaşan kısmını görünmez alana taşırıp bu testi
    # (calibrated == uncalibrated) YANLIŞ pozitif geçirebiliyordu.
    base = np.zeros((60, 220, 3), dtype=np.uint8)

    # mm_per_px=1.0 -> 10px * 1.0mm/px = 10mm = 1.00cm; sadece çökmediğini ve mm_per_px<=0
    # (px) yolundan FARKLI bir görüntü ürettiğini doğrular (metin OCR olmadan doğrudan
    # sınanamaz, ama iki yol da aynı piksel çıktısını üretmemeli).
    calibrated = draw_measurements_overlay(base, [measurement], 1.0)
    uncalibrated = draw_measurements_overlay(base, [measurement], 0.0)

    assert not np.array_equal(calibrated, uncalibrated)


def test_mm_per_px_zero_by_default_produces_no_mm_fields():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    m = out["measurements"][0]
    assert "area_mm2" not in m
    assert "bbox_mm_w" not in m


def test_mm_per_px_adds_real_world_measurements():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255  # 10x10 kare, alan=100 px

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0, "mm_per_px": 0.5})

    m = out["measurements"][0]
    assert m["area_mm2"] == m["area"] * 0.25
    assert m["bbox_mm_w"] == m["bbox_w"] * 0.5
    assert m["bbox_mm_h"] == m["bbox_h"] * 0.5
    assert m["perimeter_mm"] == m["perimeter"] * 0.5
    assert m["centroid_mm_x"] == m["centroid_x"] * 0.5
    assert m["centroid_mm_y"] == m["centroid_y"] * 0.5
    # px alanları hâlâ mevcut olmalı (geriye uyumluluk)
    assert m["area"] == 100.0


def test_obb_matches_axis_aligned_bbox_for_unrotated_square():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    m = out["measurements"][0]
    assert sorted([round(m["obb_w"]), round(m["obb_h"])]) == [9, 9]


def test_obb_is_tighter_than_bbox_for_rotated_region():
    img = np.zeros((60, 60), dtype=np.uint8)
    box = cv2.boxPoints(((30.0, 30.0), (40.0, 15.0), 30.0))
    cv2.fillPoly(img, [np.intp(box)], 255)

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    m = out["measurements"][0]
    # Döndürülmüş bir dikdörtgen için eksene-hizalı bbox, gerçek boyuttan büyük ölçer;
    # minAreaRect tabanlı oriented bbox gerçek 40x15 boyutuna yakın kalmalı.
    assert sorted([round(m["obb_w"]), round(m["obb_h"])]) != sorted([m["bbox_w"], m["bbox_h"]])
    assert m["bbox_w"] * m["bbox_h"] > m["obb_w"] * m["obb_h"]
    assert abs(sorted([m["obb_w"], m["obb_h"]])[0] - 15) < 3
    assert abs(sorted([m["obb_w"], m["obb_h"]])[1] - 40) < 3


def test_obb_mm_fields_present_when_mm_per_px_set():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0, "mm_per_px": 0.5})

    m = out["measurements"][0]
    assert m["obb_mm_w"] == m["obb_w"] * 0.5
    assert m["obb_mm_h"] == m["obb_h"] * 0.5


def test_tolerance_disabled_by_default_object_still_detected_and_measured():
    # Referans/tolerans hiç girilmeden (varsayılan tolerance_enabled=False) cisim yine
    # tespit edilip yeşil çerçeve + boyut etiketiyle ölçülmeli.
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run({"labels": _labels_for(img)}, {"min_area": 0})

    m = out["measurements"][0]
    assert "tolerance_ok" not in m
    assert m["obb_w"] > 0 and m["obb_h"] > 0
    assert (out["overlay"][5, 5:15] == [0, 255, 0]).all(axis=-1).any()


def test_tolerance_ok_true_within_range():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run(
        {"labels": _labels_for(img)},
        {
            "min_area": 0,
            "tolerance_enabled": True,
            "tol_short_min": 5,
            "tol_short_max": 12,
            "tol_long_min": 5,
            "tol_long_max": 12,
        },
    )

    m = out["measurements"][0]
    assert m["tolerance_ok"] is True
    assert (out["overlay"][5, 5:15] == [0, 255, 0]).all(axis=-1).any()


def test_tolerance_ok_false_out_of_range():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    out = RegionPropsOp().run(
        {"labels": _labels_for(img)},
        {
            "min_area": 0,
            "tolerance_enabled": True,
            "tol_short_min": 0,
            "tol_short_max": 3,  # obb kısa kenar ~9, sınırı aşıyor
            "tol_long_min": 0,
            "tol_long_max": 12,
        },
    )

    m = out["measurements"][0]
    assert m["tolerance_ok"] is False
    assert (out["overlay"][5, 5:15] == [0, 0, 255]).all(axis=-1).any()


def test_tolerance_uses_short_long_not_raw_wh():
    # 40x15 döndürülmüş dikdörtgen: obb_w/obb_h açıya göre rastgele atanabilir, ama
    # kısa/uzun kenar sıralaması açıdan bağımsız olmalı.
    img = np.zeros((60, 60), dtype=np.uint8)
    box = cv2.boxPoints(((30.0, 30.0), (40.0, 15.0), 30.0))
    cv2.fillPoly(img, [np.intp(box)], 255)

    out = RegionPropsOp().run(
        {"labels": _labels_for(img)},
        {
            "min_area": 0,
            "tolerance_enabled": True,
            "tol_short_min": 10,
            "tol_short_max": 20,
            "tol_long_min": 35,
            "tol_long_max": 45,
        },
    )

    m = out["measurements"][0]
    assert m["tolerance_ok"] is True


def test_tolerance_zero_bound_means_unbounded():
    img = np.zeros((20, 20), dtype=np.uint8)
    img[5:15, 5:15] = 255

    # Sadece tol_long_max ayarlanmış (çok dar), diğer 3 sınır varsayılan 0 (sınırsız).
    out = RegionPropsOp().run(
        {"labels": _labels_for(img)},
        {"min_area": 0, "tolerance_enabled": True, "tol_long_max": 5},
    )
    assert out["measurements"][0]["tolerance_ok"] is False

    # Aynı ayar ama gerçek boyutu geçecek kadar geniş: diğer sınırların 0/sınırsız kalması
    # her zaman geçmelerine neden olmalı, sadece ayarlanan sınır etkili olmalı.
    out = RegionPropsOp().run(
        {"labels": _labels_for(img)},
        {"min_area": 0, "tolerance_enabled": True, "tol_long_max": 20},
    )
    assert out["measurements"][0]["tolerance_ok"] is True
