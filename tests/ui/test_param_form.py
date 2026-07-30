from PySide6.QtWidgets import QCheckBox, QComboBox, QDialog, QFormLayout, QLineEdit, QPushButton, QSlider

from imgflow.core.params import ParamSpec, ParamType
from imgflow.ui.widgets.param_form import ParamForm


def _container_for(form, name):
    for i, spec in enumerate(form._specs):
        if spec.name == name:
            return form._layout.itemAt(i, QFormLayout.ItemRole.FieldRole).widget()
    return None


def test_advanced_params_are_collapsed_by_default_but_still_addressable(qtbot):
    """Gerçek kullanıcı raporu: "aşağıdaki özellikler çok karışık, daha basite indirgeyebilir
    miyiz". `advanced=True` alanlar KAPALI bir bölüme gider; alan kaybolmaz -- `values()` ve
    `widget_for()` her durumda çalışır (reçete/kaydetme davranışı DEĞİŞMEZ)."""
    form = ParamForm()
    qtbot.addWidget(form)
    specs = [
        ParamSpec("temel", ParamType.INT, default=1, min=0, max=10, label="Temel"),
        ParamSpec("ileri", ParamType.INT, default=2, min=0, max=10, label="İleri", advanced=True),
    ]

    form.set_params(specs, {"temel": 1, "ileri": 2})
    form.show()

    assert form._advanced_box.isVisible()
    assert not form._advanced_box.isChecked()
    assert form.widget_for("temel").isVisible()
    assert not form.widget_for("ileri").isVisible()  # katlanmış
    assert form.values() == {"temel": 1, "ileri": 2}  # ama değer HER ZAMAN erişilebilir

    form._advanced_box.setChecked(True)
    assert form.widget_for("ileri").isVisible()


def test_form_without_advanced_params_has_no_advanced_section(qtbot):
    """`advanced` hiç kullanılmayan spec listelerinde (ör. `camera_settings_panel.py`'nin
    dinamik GenICam alanları) düzen eskisiyle BİREBİR aynı kalmalı."""
    form = ParamForm()
    qtbot.addWidget(form)

    form.set_params([ParamSpec("a", ParamType.INT, default=1, min=0, max=5)], {"a": 1})
    form.show()

    assert not form._advanced_box.isVisible()


def test_int_param_reflects_default_and_emits_on_change(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})

    received = []
    form.params_changed.connect(received.append)
    form.widget_for("value").setValue(50)

    assert received == [{"value": 50}]
    assert form.values() == {"value": 50}


def test_enum_param_emits_selected_choice(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    specs = [ParamSpec("mode", ParamType.ENUM, default="BINARY", choices=["BINARY", "OTSU"])]
    form.set_params(specs, {"mode": "BINARY"})

    received = []
    form.params_changed.connect(received.append)
    form.widget_for("mode").setCurrentText("OTSU")

    assert received == [{"mode": "OTSU"}]


def test_bool_param_emits_toggle(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("flag", ParamType.BOOL, default=False)], {"flag": False})

    received = []
    form.params_changed.connect(received.append)
    form.widget_for("flag").setChecked(True)

    assert received == [{"flag": True}]


def test_set_params_replaces_previous_form(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("a", ParamType.INT, default=1)], {"a": 1})
    form.set_params([ParamSpec("b", ParamType.INT, default=2)], {"b": 2})

    assert form.widget_for("a") is None
    assert form.widget_for("b") is not None
    assert form.values() == {"b": 2}


def test_int_param_with_min_max_shows_synced_slider(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})

    container = _container_for(form, "value")
    slider = container.findChild(QSlider)
    assert slider is not None
    assert (slider.minimum(), slider.maximum(), slider.value()) == (0, 255, 127)

    received = []
    form.params_changed.connect(received.append)
    slider.setValue(200)

    assert form.widget_for("value").value() == 200
    assert received[-1] == {"value": 200}


def test_int_param_without_max_has_no_slider(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("iterations", ParamType.INT, default=1, min=1)], {"iterations": 1})

    container = _container_for(form, "iterations")
    assert container.findChild(QSlider) is None


def test_float_param_slider_scales_and_syncs(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("sigma_x", ParamType.FLOAT, default=0.0, min=0.0, max=50.0)], {"sigma_x": 0.0})

    container = _container_for(form, "sigma_x")
    slider = container.findChild(QSlider)
    assert slider is not None

    received = []
    form.params_changed.connect(received.append)
    slider.setValue(2550)  # 25.50 * 100 ölçek

    assert abs(form.widget_for("sigma_x").value() - 25.5) < 1e-6
    assert abs(received[-1]["sigma_x"] - 25.5) < 1e-6


def test_field_changed_emits_name_and_value(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})

    received = []
    form.field_changed.connect(lambda name, value: received.append((name, value)))
    form.widget_for("value").setValue(50)

    assert received == [("value", 50)]


def test_readonly_param_disables_container(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255, readonly=True)], {"value": 127})

    container = _container_for(form, "value")
    assert container.isEnabled() is False


def test_set_value_updates_widget_without_emitting_signals(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})

    params_received = []
    field_received = []
    form.params_changed.connect(params_received.append)
    form.field_changed.connect(lambda name, value: field_received.append((name, value)))

    form.set_value("value", 200)

    assert form.widget_for("value").value() == 200
    assert form.values()["value"] == 200
    assert params_received == []
    assert field_received == []


def test_set_value_keeps_slider_in_sync(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})
    container = _container_for(form, "value")
    slider = container.findChild(QSlider)

    form.set_value("value", 200)

    assert slider.value() == 200


def test_set_value_on_enum_widget(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    specs = [ParamSpec("mode", ParamType.ENUM, default="BINARY", choices=["BINARY", "OTSU"])]
    form.set_params(specs, {"mode": "BINARY"})

    received = []
    form.field_changed.connect(lambda *args: received.append(args))
    form.set_value("mode", "OTSU")

    assert form.widget_for("mode").currentText() == "OTSU"
    assert received == []


def test_spinbox_change_updates_slider_position(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("value", ParamType.INT, default=127, min=0, max=255)], {"value": 127})

    container = _container_for(form, "value")
    slider = container.findChild(QSlider)
    form.widget_for("value").setValue(10)

    assert slider.value() == 10


def test_string_param_without_dynamic_choices_uses_plain_line_edit(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    form.set_params([ParamSpec("text", ParamType.STRING, default="hello")], {"text": "hello"})

    assert isinstance(form.widget_for("text"), QLineEdit)


def test_string_param_with_dynamic_choices_renders_populated_combo(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec("model_name", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"])
    form.set_params([spec], {"model_name": "a"})

    widget = form.widget_for("model_name")
    assert isinstance(widget, QComboBox)
    assert [widget.itemText(i) for i in range(widget.count())] == ["a", "b"]
    assert widget.currentText() == "a"


def test_string_param_with_dynamic_choices_keeps_unsaved_value_selected(qtbot):
    """Kayıtlı listede henüz olmayan (ör. serbest yazılmış) bir değer sessizce kaybolmamalı."""
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec("model_name", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"])
    form.set_params([spec], {"model_name": "henuz_kayitli_degil"})

    widget = form.widget_for("model_name")
    assert widget.currentText() == "henuz_kayitli_degil"


def test_string_param_with_dynamic_choices_emits_on_change(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec("model_name", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"])
    form.set_params([spec], {"model_name": "a"})

    received = []
    form.params_changed.connect(received.append)
    form.widget_for("model_name").setCurrentText("b")

    assert received == [{"model_name": "b"}]


def test_dynamic_choice_combo_autoselect_syncs_value_when_stored_value_is_empty(qtbot):
    # Gerçek kullanıcı raporu: kayıtlı bir referans/model varken (ör. "deneme1"), boş
    # (`""`) saklanmış bir düğüm parametresiyle form kurulunca Qt, QComboBox'a `addItems()`
    # ile öğe eklenir eklenmez OTOMATİK olarak ilk öğeyi ("deneme1") seçili gösteriyordu --
    # ama bu, sinyal bağlantısından ÖNCE olduğu için `self._values`'e hiç yansımıyordu.
    # Sonuç: ekranda dolu görünen bir alan aslında hâlâ boştu, kullanıcı hiç dokunmasa bile
    # operatör "parametre boş olamaz" hatası vermeye devam ediyordu.
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec("reference_name", ParamType.STRING, default="", dynamic_choices=lambda: ["deneme1", "b"])

    received = []
    form.params_changed.connect(received.append)
    form.set_params([spec], {"reference_name": ""})

    widget = form.widget_for("reference_name")
    assert widget.currentText() == "deneme1"
    # Widget'ın gösterdiği değerle `_values`/dışarı yayılan değer artık SENKRON olmalı.
    assert form.values()["reference_name"] == "deneme1"
    assert received == [{"reference_name": "deneme1"}]


def test_dynamic_choice_combo_no_spurious_emit_when_value_already_matches(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec("model_name", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"])

    received = []
    form.params_changed.connect(received.append)
    form.set_params([spec], {"model_name": "a"})

    assert received == []  # zaten senkron -- gereksiz bir "düzeltme" sinyali yayılmamalı


def test_enum_combo_autoselect_syncs_value_when_stored_value_not_in_choices(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    specs = [ParamSpec("mode", ParamType.ENUM, default="BINARY", choices=["BINARY", "OTSU"])]

    received = []
    form.params_changed.connect(received.append)
    # Eski bir reçetede/kayıtta artık geçerli olmayan bir değer (ör. kaldırılmış bir seçenek).
    form.set_params(specs, {"mode": "ARTIK_GECERSIZ"})

    assert form.widget_for("mode").currentText() == "BINARY"
    assert form.values()["mode"] == "BINARY"
    assert received == [{"mode": "BINARY"}]


def test_multi_select_widget_is_editable_line_edit_showing_current_value(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec(
        "model_names", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"], multi_select=True
    )
    form.set_params([spec], {"model_names": "a"})

    widget = form.widget_for("model_names")
    assert isinstance(widget, QLineEdit)
    assert not widget.isReadOnly()
    assert widget.text() == "a"


def test_multi_select_widget_allows_typing_an_unregistered_name_manually(qtbot):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec(
        "model_names", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"], multi_select=True
    )
    form.set_params([spec], {"model_names": ""})

    received = []
    form.params_changed.connect(received.append)
    form.widget_for("model_names").setText("henuz_kaydedilmemis_model")

    assert received == [{"model_names": "henuz_kaydedilmemis_model"}]


def test_multi_select_picker_confirms_current_selection_unchanged(qtbot, monkeypatch):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec(
        "model_names", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b", "c"], multi_select=True
    )
    form.set_params([spec], {"model_names": "a, c"})
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Accepted)

    received = []
    form.params_changed.connect(received.append)
    container = _container_for(form, "model_names")
    container.findChild(QPushButton).click()

    assert form.widget_for("model_names").text() == "a, c"
    assert received == [{"model_names": "a, c"}]


def test_multi_select_picker_cancel_keeps_previous_value(qtbot, monkeypatch):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec(
        "model_names", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"], multi_select=True
    )
    form.set_params([spec], {"model_names": "a"})
    monkeypatch.setattr(QDialog, "exec", lambda self: QDialog.DialogCode.Rejected)

    received = []
    form.params_changed.connect(received.append)
    container = _container_for(form, "model_names")
    container.findChild(QPushButton).click()

    assert form.widget_for("model_names").text() == "a"
    assert received == []


def test_multi_select_picker_reflects_checkbox_changes_made_before_accept(qtbot, monkeypatch):
    form = ParamForm()
    qtbot.addWidget(form)
    spec = ParamSpec(
        "model_names", ParamType.STRING, default="", dynamic_choices=lambda: ["a", "b"], multi_select=True
    )
    form.set_params([spec], {"model_names": "a"})

    def _accept_with_b_checked(dialog_self):
        for checkbox in dialog_self.findChildren(QCheckBox):
            if checkbox.text() == "b":
                checkbox.setChecked(True)
        return QDialog.DialogCode.Accepted

    monkeypatch.setattr(QDialog, "exec", _accept_with_b_checked)
    container = _container_for(form, "model_names")
    container.findChild(QPushButton).click()

    assert form.widget_for("model_names").text() == "a, b"
