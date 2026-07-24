from imgflow.ui.dialogs.help_dialog import HelpDialog


def test_help_dialog_shows_calibration_workflow_sections(qtbot):
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    browser = dialog.layout().itemAt(0).widget()
    html = browser.toPlainText()

    assert "Kamera Ayarları" in html
    assert "Lens Kalibrasyonu" in html
    assert "Yükseklik-Ölçek Kalibrasyonu" in html
    assert "Dijital I/O" in html
    assert "Aktarım Katmanı" in html
    assert "Kullanıcı Setleri" in html


def test_help_dialog_is_non_modal(qtbot):
    dialog = HelpDialog()
    qtbot.addWidget(dialog)

    assert dialog.isModal() is False
