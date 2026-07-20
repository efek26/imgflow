from imgflow.core.params import ParamSpec, ParamType
from imgflow.ui.widgets.param_form import ParamForm


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
