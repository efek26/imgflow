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


def test_measurements_without_model_key_shows_generic_total(qtbot):
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([{"area": 100.0}, {"area": 50.0}])

    assert panel.text() == "Toplam ölçüm: 2"


def test_single_row_measurement_without_model_lists_key_values(qtbot):
    """`analysis.color_props`/`analysis.texture_props` gibi tüm-görüntü için TEK satır
    üreten operatörlerin çıktısı, sayı yerine okunabilir 'anahtar: değer' satırları olarak
    gösterilmeli."""
    panel = MeasurementsSummaryPanel()
    qtbot.addWidget(panel)

    panel.set_measurements([{"l_mean": 50.234, "a_mean": 0.1, "class_name": "kusur"}])

    text = panel.text()
    assert "l_mean: 50.23" in text
    assert "a_mean: 0.10" in text
    assert "class_name: kusur" in text


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
