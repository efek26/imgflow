from PySide6.QtWidgets import QLabel, QPushButton

from imgflow.core.camera_params import (
    CATEGORIES,
    CATEGORY_ACQUISITION,
    CATEGORY_IMAGE_QUALITY,
    CATEGORY_USER_SETS,
    CameraParameterController,
)
from imgflow.ui.panels.camera_settings_panel import CameraSettingsPanel
from tests.support.fake_genicam import FakeGenicamNode, FakeNodeMap, default_camera_nodes


def _controller(**overrides) -> CameraParameterController:
    return CameraParameterController(FakeNodeMap(default_camera_nodes(**overrides)))


def test_set_controller_none_shows_disconnected_status(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)

    assert "bağlı değil" in panel._status_label.text()
    assert panel._toolbox.count() == 0


def test_set_controller_builds_toolbox_pages_with_icons(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)

    panel.set_controller(_controller(), status_text="🟢 Bağlı (test)")

    assert panel._status_label.text() == "🟢 Bağlı (test)"
    labels = [panel._toolbox.itemText(i) for i in range(panel._toolbox.count())]
    assert any(CATEGORY_ACQUISITION in label for label in labels)
    assert any("⏱️" in label for label in labels)


def test_set_controller_none_clears_previous_pages(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(_controller())
    assert panel._toolbox.count() > 0

    panel.set_controller(None)

    assert panel._toolbox.count() == 0
    assert panel._forms == {}


def test_user_sets_page_shows_save_command_button(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(_controller())

    labels = [panel._toolbox.itemText(i) for i in range(panel._toolbox.count())]
    user_sets_index = next(i for i, label in enumerate(labels) if CATEGORY_USER_SETS in label)
    page = panel._toolbox.widget(user_sets_index)
    buttons = [b for b in page.findChildren(QPushButton) if "Kaydet" in b.text()]
    assert len(buttons) == 1


def test_every_category_page_shows_non_empty_description(qtbot):
    from imgflow.core.camera_params import CameraParameterController
    from tests.support.fake_genicam import FakeGenicamNode, FakeNodeMap, default_camera_nodes

    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    # Aktarım Katmanı düğümleri varsayılan sahte node-map'te available=False (gerçek GigE
    # kamerada olduğu gibi USB3'te gizli olabilirler) — bu testte sayfanın var olduğunu
    # doğrulamak için o kategoriden en az bir node'u kullanılabilir kılıyoruz.
    nodes = default_camera_nodes(
        GevSCPSPacketSize=FakeGenicamNode(1500, min=220, max=9000, available=True)
    )
    controller = CameraParameterController(FakeNodeMap(nodes))
    panel.set_controller(controller)

    labels = [panel._toolbox.itemText(i) for i in range(panel._toolbox.count())]
    for category in CATEGORIES:
        matching = [i for i, label in enumerate(labels) if category in label]
        assert matching, f"{category} sayfası bulunamadı"
        page = panel._toolbox.widget(matching[0])
        description_labels = [
            w for w in page.findChildren(QLabel) if "font-style: italic" in w.styleSheet()
        ]
        assert len(description_labels) == 1
        assert description_labels[0].text().strip() != ""


def test_field_change_is_debounced_and_not_applied_immediately(qtbot):
    controller = _controller()
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    form = panel._forms[CATEGORY_ACQUISITION]
    form.widget_for("AcquisitionFrameRate").setValue(60.0)

    # Henüz debounce süresi geçmediği için donanıma hiçbir şey yazılmamış olmalı
    specs = controller.build_specs(CATEGORY_ACQUISITION)
    assert controller.current_values(specs)["AcquisitionFrameRate"] == 30.0  # varsayılan, henüz değişmedi
    assert "AcquisitionFrameRate" in panel._pending_changes


def test_debounce_timer_eventually_applies_change(qtbot):
    controller = _controller()
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    form = panel._forms[CATEGORY_ACQUISITION]
    form.widget_for("AcquisitionFrameRate").setValue(60.0)

    qtbot.wait(300)  # debounce süresinin (120ms) geçmesini bekle

    specs = controller.build_specs(CATEGORY_ACQUISITION)
    assert controller.current_values(specs)["AcquisitionFrameRate"] == 60.0


def test_non_structural_change_does_not_rebuild_category(qtbot):
    controller = _controller()
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    calls = []
    original_build_specs = controller.build_specs
    controller.build_specs = lambda *a, **kw: (calls.append(1), original_build_specs(*a, **kw))[1]

    form = panel._forms[CATEGORY_ACQUISITION]
    form.widget_for("AcquisitionFrameRate").setValue(60.0)  # ExposureTime/Gain gibi yapısal olmayan
    panel._flush_pending_changes()

    assert calls == []  # kategori yeniden kurulmadı, tek alan güncellendi


def test_structural_change_rebuilds_category():
    controller = _controller()
    panel = CameraSettingsPanel()
    panel.set_controller(controller)

    calls = []
    original_build_specs = controller.build_specs
    controller.build_specs = lambda *a, **kw: (calls.append(1), original_build_specs(*a, **kw))[1]

    form = panel._forms[CATEGORY_ACQUISITION]
    form.widget_for("TriggerMode").setCurrentText("On")  # affects_availability=True
    panel._flush_pending_changes()

    assert len(calls) >= 1


def test_execute_command_calls_controller(qtbot):
    controller = _controller()
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    executed = []
    controller.execute = lambda name: executed.append(name)

    panel._on_execute("UserSetSave")

    assert executed == ["UserSetSave"]


def test_refresh_requeries_current_controller(qtbot):
    controller = _controller()
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller, status_text="🟢 Bağlı (test)")

    # Donanım tarafında değer harici olarak değişmiş gibi davran (ör. Pylon Viewer'dan)
    controller._node_map.get_node("Gain").set_value(5.0)
    panel.refresh()

    # Gain "Görüntü Kalitesi" kategorisinde
    quality_form = panel._forms[CATEGORY_IMAGE_QUALITY]
    assert quality_form.widget_for("Gain").value() == 5.0


def test_hardware_write_failure_shows_inline_error_and_does_not_crash(qtbot):
    controller = _controller(Gain=FakeGenicamNode(0.0, min=0.0, max=24.0, writable=False))
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    form = panel._forms[CATEGORY_IMAGE_QUALITY]
    form.widget_for("Gain").setValue(5.0)
    panel._flush_pending_changes()  # patlamamalı (önceden burada yakalanmayan RuntimeError fırlıyordu)

    assert panel._error_label.text() != ""


def test_hardware_write_failure_never_shows_a_modal_dialog(qtbot, monkeypatch):
    """Kritik regresyon testi: bu yol bir QTimer debounce'u üzerinden çalışıyor — modal bir
    QMessageBox burada gösterilirse ve alan sürekli reddedilirse (ör. TriggerMode) her
    tetiklemede yeni bir modal açılıp kullanıcı uygulamayı kapatamaz hale gelebilir (gerçek
    bir kullanıcı raporuyla doğrulandı). Bu yüzden hiçbir QMessageBox çağrılmamalı."""
    controller = _controller(Gain=FakeGenicamNode(0.0, min=0.0, max=24.0, writable=False))
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    calls = []
    monkeypatch.setattr("PySide6.QtWidgets.QMessageBox.critical", lambda *a, **k: calls.append(1))

    form = panel._forms[CATEGORY_IMAGE_QUALITY]
    for value in (5.0, 6.0, 7.0):
        form.widget_for("Gain").setValue(value)
        panel._flush_pending_changes()

    assert calls == []


def test_execute_failure_shows_inline_error_and_does_not_crash(qtbot):
    controller = _controller(UserSetSave=FakeGenicamNode(False, writable=False))
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    panel._on_execute("UserSetSave")  # patlamamalı

    assert panel._error_label.text() != ""


def test_save_load_buttons_disabled_without_controller(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)

    assert not panel._save_settings_button.isEnabled()
    assert not panel._load_settings_button.isEnabled()


def test_save_load_buttons_enabled_with_controller(qtbot):
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(_controller())

    assert panel._save_settings_button.isEnabled()
    assert panel._load_settings_button.isEnabled()


def test_save_settings_writes_current_values_to_store(qtbot, tmp_path, monkeypatch):
    controller = _controller()
    panel = CameraSettingsPanel(camera_settings_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    monkeypatch.setattr(
        "imgflow.ui.panels.camera_settings_panel.QInputDialog.getItem",
        lambda *a, **k: ("hat1", True),
    )

    panel._on_save_settings()

    assert panel._io_status_label.text() == "Kaydedildi: 'hat1'"
    from imgflow.io_utils import camera_settings_store

    assert camera_settings_store.list_settings(tmp_path) == ["hat1"]
    saved = camera_settings_store.load_settings("hat1", directory=tmp_path)
    assert saved.values["Gain"] == 0.0


def test_load_settings_applies_values_and_rebuilds_panel(qtbot, tmp_path, monkeypatch):
    from imgflow.io_utils import camera_settings_store

    camera_settings_store.save_settings("hat1", {"Gain": 7.5}, directory=tmp_path)

    controller = _controller()
    panel = CameraSettingsPanel(camera_settings_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    monkeypatch.setattr(
        "imgflow.ui.panels.camera_settings_panel.QInputDialog.getItem",
        lambda *a, **k: ("hat1", True),
    )

    panel._on_load_settings()

    quality_form = panel._forms[CATEGORY_IMAGE_QUALITY]
    assert quality_form.widget_for("Gain").value() == 7.5
    assert "Yüklendi: 'hat1'" in panel._io_status_label.text()


def test_load_settings_reports_no_saved_presets(qtbot, tmp_path):
    controller = _controller()
    panel = CameraSettingsPanel(camera_settings_dir=tmp_path)
    qtbot.addWidget(panel)
    panel.set_controller(controller)

    panel._on_load_settings()

    assert panel._io_status_label.text() == "Kayıtlı kamera ayarı ön ayarı yok."


def test_successful_change_clears_previous_error(qtbot):
    controller = _controller(Gain=FakeGenicamNode(0.0, min=0.0, max=24.0, writable=False))
    panel = CameraSettingsPanel()
    qtbot.addWidget(panel)
    panel.set_controller(controller)
    form = panel._forms[CATEGORY_IMAGE_QUALITY]
    form.widget_for("Gain").setValue(5.0)
    panel._flush_pending_changes()
    assert panel._error_label.text() != ""

    controller._node_map.get_node("Gain")._writable = True
    form.widget_for("Gain").setValue(6.0)
    panel._flush_pending_changes()

    assert panel._error_label.text() == ""
