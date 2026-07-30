from imgflow.ui.widgets.measurements_summary import MeasurementsSummaryPanel


def test_empty_measurements_shows_placeholder(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([])

    assert "yok" in panel.text().lower()


def test_none_measurements_shows_placeholder(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(None)

    assert "yok" in panel.text().lower()


def test_measurements_with_model_key_lists_each_match_by_number(qtbot):
    """Kullanıcı isteği: "her şekili 1,2,3,4 diye adlandıralım, sonra 1'in x,y,alpha
    değerleri... yazsın" — panel artık model başına SAYIM değil, her eşleşmeyi numarasıyla
    ve x/y/alpha değerleriyle tek tek listeler."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {"model": "civata", "label": "1", "x": 10.0, "y": 20.0, "angle": 30.0, "score": 0.9},
            {"model": "somun", "label": "2", "x": 40.0, "y": 50.0, "angle": -15.0, "score": 0.8},
        ]
    )

    text = panel.text()
    assert "1 (civata): x=10.0  y=20.0  α=30.0°" in text
    assert "2 (somun): x=40.0  y=50.0  α=-15.0°" in text
    assert "Toplam: 2" in text


def test_shape_match_measurements_show_size_and_displacement_in_cm_when_calibrated(qtbot):
    """Gerçek kullanıcı isteği: "geometrik eşlemede scale boyutu ve cismin ötelenme uzaklığı
    da yazmalı", sonraki turlarda netleşti: "ötelemeyi x,y şeklinde yaz ve kalibrasyon varsa
    cm şeklinde yaz" + "kalibrasyon seçili olduğu her senaryoda pixelin yanında mm de yazsın
    veya cm" -- kalibrasyon varsa hem öteleme (cm) hem boyut (px değeri de mm'nin YANINDA,
    gizlenmeden) gösterilir."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "model": "civata",
                "label": "1",
                "x": 10.0,
                "y": 20.0,
                "angle": 30.0,
                "score": 0.9,
                "width_px": 50.0,
                "height_px": 25.0,
                "width_mm": 25.0,
                "height_mm": 12.5,
                "displacement_px": 14.14,
                "displacement_x_px": -10.0,
                "displacement_y_px": 10.0,
                "displacement_cm": 0.707,
                "displacement_x_cm": -0.5,
                "displacement_y_cm": 0.5,
            }
        ]
    )

    text = panel.text()
    assert "boy=50x25px (25.0x12.5mm)" in text
    assert "öteleme=x:-0.50 y:+0.50cm" in text


def test_shape_match_measurements_show_displacement_xy_in_px_without_calibration(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "model": "civata",
                "label": "1",
                "x": 10.0,
                "y": 20.0,
                "angle": 30.0,
                "score": 0.9,
                "displacement_px": 14.14,
                "displacement_x_px": -10.0,
                "displacement_y_px": 10.0,
            }
        ]
    )

    text = panel.text()
    assert "öteleme=x:-10.0 y:+10.0px" in text
    assert "boy=" not in text


def test_measurements_without_model_key_shows_generic_total(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([{"area": 100.0}, {"area": 50.0}])

    assert panel.text() == "Toplam ölçüm: 2"


def test_single_row_measurement_without_model_lists_key_values(qtbot):
    """`analysis.color_props`/`analysis.texture_props` gibi tüm-görüntü için TEK satır
    üreten operatörlerin çıktısı, ham anahtar adı yerine açıklamalı bir etiketle ('L*
    (Parlaklık) Ort.: 50.23' gibi) gösterilmeli — bilinmeyen anahtarlar (ör. class_name)
    olduğu gibi kalır."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([{"l_mean": 50.234, "a_mean": 0.1, "class_name": "kusur"}])

    text = panel.text()
    assert "L* (Parlaklık) Ort.: 50.23" in text
    assert "a* (Yeşil↔Kırmızı) Ort.: 0.10" in text
    assert "class_name: kusur" in text


def test_per_object_color_measurements_lists_each_object_by_number(qtbot):
    """`analysis.color_props`'un "Otomatik Nesne Tespiti" açıkken ürettiği çıktı: her nesne
    numarasıyla ve açıklamalı L/a/b etiketleriyle listelenmeli."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {"label": 1, "l_mean": 60.0, "a_mean": 20.0, "b_mean": -5.0, "l_std": 1.0, "a_std": 1.0, "b_std": 1.0},
            {
                "label": 2,
                "l_mean": 40.0,
                "a_mean": -10.0,
                "b_mean": 15.0,
                "l_std": 1.0,
                "a_std": 1.0,
                "b_std": 1.0,
                "delta_e": 8.5,
                "tolerance_ok": False,
            },
        ]
    )

    text = panel.text()
    assert "#1 → L(parlaklık)=60.0  a(yeşil↔kırmızı)=20.0  b(mavi↔sarı)=-5.0" in text
    assert "#2 → L(parlaklık)=40.0  a(yeşil↔kırmızı)=-10.0  b(mavi↔sarı)=15.0  ΔE=8.5 (NG)" in text
    assert "Toplam nesne: 2" in text


def test_per_object_texture_measurements_lists_each_object_by_number(qtbot):
    """`analysis.texture_props`'un "Otomatik Nesne Tespiti" açıkken ürettiği çıktı: her
    nesne numarasıyla ve açıklamalı Kontrast/Homojenlik/Enerji/Korelasyon etiketleriyle
    listelenmeli."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {"label": 1, "contrast": 2.5, "homogeneity": 0.8, "energy": 0.3, "correlation": 0.6},
        ]
    )

    text = panel.text()
    assert "#1 → Kontrast=2.50  Homojenlik=0.80  Enerji=0.30  Korelasyon=0.60" in text
    assert "Toplam nesne: 1" in text


def test_manual_roi_color_measurements_tagged_roi_with_range(qtbot):
    """`analysis.color_props`'un "Elle ROI Çiz" modu: `manual=True` etiketli satırlar "#N"
    yerine "ROIN" diye numaralanmalı ve L/a/b min-max ARALIĞI ayrı bir satırda gösterilmeli
    (gerçek kullanıcı isteği: "roi içindeki renk dalgalanmalarını aralık olarak versin")."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "label": 1,
                "manual": True,
                "l_mean": 60.0,
                "a_mean": 20.0,
                "b_mean": -5.0,
                "l_std": 5.0,
                "a_std": 1.0,
                "b_std": 1.0,
                "l_min": 40.0,
                "l_max": 80.0,
                "a_min": 15.0,
                "a_max": 25.0,
                "b_min": -10.0,
                "b_max": 0.0,
                "bbox_x": 0,
                "bbox_y": 0,
                "bbox_w": 10,
                "bbox_h": 10,
            },
        ]
    )

    text = panel.text()
    assert "ROI1 → L(parlaklık)=60.0  a(yeşil↔kırmızı)=20.0  b(mavi↔sarı)=-5.0" in text
    assert "Aralık (dalgalanma): L[40.0,80.0]  a[15.0,25.0]  b[-10.0,0.0]" in text
    assert "Toplam nesne: 1" in text


def test_manual_roi_texture_measurements_tagged_roi(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "label": 1,
                "manual": True,
                "contrast": 2.5,
                "homogeneity": 0.8,
                "energy": 0.3,
                "correlation": 0.6,
            },
        ]
    )

    text = panel.text()
    assert "ROI1 → Kontrast=2.50  Homojenlik=0.80  Enerji=0.30  Korelasyon=0.60" in text


def test_measurements_with_confidence_key_lists_each_detection_by_number(qtbot):
    """`ml.onnx_detect`'in çıktısı: 'model' dalıyla AYNI mantık ama x/y/alpha yerine sınıf
    adı + güven skoru + kutu."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "model": "kusur_dedektoru",
                "label": "1",
                "class_name": "kusur",
                "confidence": 0.873,
                "bbox_x": 10.0,
                "bbox_y": 20.0,
                "bbox_w": 30.0,
                "bbox_h": 40.0,
            }
        ]
    )

    text = panel.text()
    assert "1 (kusur): %87  [x=10 y=20 w=30 h=40]" in text
    assert "Toplam: 1" in text


def test_onnx_measurements_without_tolerance_ok_unchanged(qtbot):
    """`defect_classes` boşken (varsayılan) `tolerance_ok` alanı hiç üretilmez -- mevcut
    davranış (OK/NG etiketi/satırı YOK) korunmalı."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "model": "kusur_dedektoru",
                "label": "1",
                "class_name": "kusur",
                "confidence": 0.873,
                "bbox_x": 10.0,
                "bbox_y": 20.0,
                "bbox_w": 30.0,
                "bbox_h": 40.0,
            }
        ]
    )

    text = panel.text()
    assert "[OK]" not in text
    assert "[NG]" not in text
    assert "OK:" not in text


def test_onnx_measurements_with_tolerance_ok_shows_status_and_ok_ng_totals(qtbot):
    """`ml.onnx_detect`'in `defect_classes` parametresi doluyken ürettiği çıktı: her satırda
    [OK]/[NG] ve altında toplam OK/NG sayımı gösterilmeli (gerçek kullanıcı isteği: tespit
    edilen ürünü bozuk/hatalı diye sınıflandırıp güven faktörünü yazsın)."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {
                "model": "kusur_dedektoru",
                "label": "1",
                "class_name": "kusur",
                "confidence": 0.92,
                "bbox_x": 10.0,
                "bbox_y": 20.0,
                "bbox_w": 30.0,
                "bbox_h": 40.0,
                "tolerance_ok": False,
            },
            {
                "model": "kusur_dedektoru",
                "label": "2",
                "class_name": "saglam",
                "confidence": 0.88,
                "bbox_x": 60.0,
                "bbox_y": 20.0,
                "bbox_w": 30.0,
                "bbox_h": 40.0,
                "tolerance_ok": True,
            },
        ]
    )

    text = panel.text()
    assert "1 (kusur): %92  [x=10 y=20 w=30 h=40]  [NG]" in text
    assert "2 (saglam): %88  [x=60 y=20 w=30 h=40]  [OK]" in text
    assert "OK: 1  NG: 1" in text


def test_measurements_with_tolerance_ok_shows_ok_ng_counts(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements(
        [
            {"tolerance_ok": True, "label": 1},
            {"tolerance_ok": False, "label": 2},
        ]
    )

    text = panel.text()
    assert "OK: 1" in text
    assert "NG: 1" in text
    assert "Toplam: 2" in text


def test_measurements_updates_replace_previous_text(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([{"model": "civata", "label": "1", "x": 0.0, "y": 0.0, "angle": 0.0, "score": 1.0}])
    panel.set_measurements([])

    assert "yok" in panel.text().lower()


def test_step_durations_table_hidden_when_no_rows(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_step_durations(None)
    assert panel._duration_table.isHidden() is True
    assert panel._duration_heading.isHidden() is True

    panel.set_step_durations([])
    assert panel._duration_table.isHidden() is True


def test_step_durations_table_shows_rows_in_given_order_as_ms(qtbot):
    """Gerçek kullanıcı isteği: "her işlemin sonucunun süresi sonuçlar kısmında yazmalı"."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_step_durations([("Görüntü Yükle", 0.002), ("Şekil Eşleştirme (Bul)", 0.842)])

    assert panel._duration_table.isHidden() is False
    assert panel._duration_heading.isHidden() is False
    assert panel._duration_table.rowCount() == 2
    assert panel._duration_table.item(0, 0).text() == "Görüntü Yükle"
    assert panel._duration_table.item(0, 1).text() == "2.0 ms"
    assert panel._duration_table.item(1, 0).text() == "Şekil Eşleştirme (Bul)"
    assert panel._duration_table.item(1, 1).text() == "842.0 ms"


def test_step_durations_table_replaces_previous_rows(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_step_durations([("A", 0.001), ("B", 0.002), ("C", 0.003)])
    panel.set_step_durations([("A", 0.001)])

    assert panel._duration_table.rowCount() == 1
