import numpy as np
import pytest
from PySide6.QtWidgets import QMessageBox

from imgflow.core import custom_filters
from imgflow.ui.dialogs.custom_filter_dialog import CustomFilterDialog


@pytest.fixture(autouse=True)
def _isolated_custom_filter_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_filters, "CUSTOM_FILTER_DIR", tmp_path / "custom_filters")
    yield
    for op_id in list(custom_filters.registry_module._ops):
        if op_id.startswith(custom_filters.OP_ID_PREFIX):
            custom_filters.registry_module.unregister(op_id)


def _test_image() -> np.ndarray:
    return np.zeros((8, 8, 3), dtype=np.uint8)


def _dialog(qtbot, frame=None, is_op_in_use=None) -> CustomFilterDialog:
    dialog = CustomFilterDialog(frame_provider=lambda: frame, is_op_in_use=is_op_in_use, parent=None)
    qtbot.addWidget(dialog)
    return dialog


def test_dialog_starts_with_template_code(qtbot):
    dialog = _dialog(qtbot)
    assert "def apply(image)" in dialog._code_edit.toPlainText()


def test_test_button_shows_error_when_no_image_available(qtbot):
    dialog = _dialog(qtbot, frame=None)
    dialog._on_test()
    assert "görüntü yok" in dialog._error_label.text()


def test_test_button_runs_code_on_provided_frame(qtbot):
    dialog = _dialog(qtbot, frame=_test_image())
    dialog._name_edit.setText("Deneme")
    dialog._code_edit.setPlainText("def apply(image):\n    return image + 1\n")

    dialog._on_test()

    assert dialog._error_label.text() == ""
    assert dialog._preview._pixmap is not None


def test_test_button_shows_compile_error(qtbot):
    dialog = _dialog(qtbot, frame=_test_image())
    dialog._code_edit.setPlainText("not valid python(\n")

    dialog._on_test()

    assert "Kod hatası" in dialog._error_label.text()


def test_save_persists_filter_and_updates_list(qtbot):
    dialog = _dialog(qtbot, frame=_test_image())
    dialog._name_edit.setText("Kaydedilen Filtre")
    dialog._code_edit.setPlainText("def apply(image):\n    return image\n")

    received = []
    dialog.filters_changed.connect(lambda: received.append(1))
    dialog._on_save()

    assert "Kaydedildi" in dialog._status_label.text()
    assert received == [1]
    names = [dialog._list.item(i).text() for i in range(dialog._list.count())]
    assert "Kaydedilen Filtre" in names


def test_save_without_name_shows_warning(qtbot, monkeypatch):
    warned = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.custom_filter_dialog.QMessageBox.warning",
        lambda *a, **k: warned.append(1),
    )
    dialog = _dialog(qtbot)
    dialog._name_edit.setText("")

    dialog._on_save()

    assert warned == [1]


def test_save_with_bad_code_shows_error_and_does_not_persist(qtbot):
    dialog = _dialog(qtbot)
    dialog._name_edit.setText("Bozuk")
    dialog._code_edit.setPlainText("not valid python(\n")

    dialog._on_save()

    assert dialog._error_label.text() != ""
    assert custom_filters.list_custom_filters() == []


def test_selecting_existing_filter_loads_it_into_editor(qtbot):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Var Olan", code="def apply(image):\n    return image\n")
    )
    dialog = _dialog(qtbot)

    item = dialog._list.item(0)
    dialog._list.setCurrentItem(item)

    assert dialog._name_edit.text() == "Var Olan"
    assert "def apply(image)" in dialog._code_edit.toPlainText()


def test_new_button_clears_editor(qtbot):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Var Olan", code="def apply(image):\n    return image\n")
    )
    dialog = _dialog(qtbot)
    dialog._list.setCurrentItem(dialog._list.item(0))

    dialog._on_new()

    assert dialog._name_edit.text() == ""
    assert dialog._list.currentItem() is None


def test_delete_removes_filter_after_confirmation(qtbot, monkeypatch):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Silinecek", code="def apply(image):\n    return image\n")
    )
    monkeypatch.setattr(
        "imgflow.ui.dialogs.custom_filter_dialog.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.Yes,
    )
    dialog = _dialog(qtbot)
    dialog._list.setCurrentItem(dialog._list.item(0))

    dialog._on_delete()

    assert custom_filters.list_custom_filters() == []
    assert dialog._list.count() == 0


def test_delete_declined_keeps_filter(qtbot, monkeypatch):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Kalacak", code="def apply(image):\n    return image\n")
    )
    monkeypatch.setattr(
        "imgflow.ui.dialogs.custom_filter_dialog.QMessageBox.question",
        lambda *a, **k: QMessageBox.StandardButton.No,
    )
    dialog = _dialog(qtbot)
    dialog._list.setCurrentItem(dialog._list.item(0))

    dialog._on_delete()

    assert len(custom_filters.list_custom_filters()) == 1


def test_delete_blocked_when_filter_in_use(qtbot, monkeypatch):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Kullanımda", code="def apply(image):\n    return image\n")
    )
    warned = []
    monkeypatch.setattr(
        "imgflow.ui.dialogs.custom_filter_dialog.QMessageBox.warning",
        lambda *a, **k: warned.append(1),
    )
    dialog = _dialog(qtbot, is_op_in_use=lambda op_id: True)
    dialog._list.setCurrentItem(dialog._list.item(0))

    dialog._on_delete()

    assert warned == [1]
    assert len(custom_filters.list_custom_filters()) == 1
