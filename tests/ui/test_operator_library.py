import pytest
from PySide6.QtCore import Qt

from imgflow.core import custom_filters
from imgflow.operators import registry
from imgflow.ui.panels.operator_library import (
    UNCATEGORIZED_LABEL,
    OperatorLibrary,
    category_for,
    description_for,
    label_for,
)


@pytest.fixture
def _isolated_custom_filter_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(custom_filters, "CUSTOM_FILTER_DIR", tmp_path / "custom_filters")
    yield
    for op_id in list(registry._ops):
        if op_id.startswith(custom_filters.OP_ID_PREFIX):
            registry.unregister(op_id)


def test_category_for_known_op_id():
    assert category_for("morphology.erode") == "Morfoloji"


def test_category_for_unknown_op_id_falls_back_to_uncategorized():
    assert category_for("no.such.operator") == UNCATEGORIZED_LABEL


def test_label_and_description_fall_back_to_op_id_when_unmapped():
    assert label_for("no.such.operator") == "no.such.operator"
    assert description_for("no.such.operator") == "no.such.operator"


def test_refresh_populates_tree_grouped_by_category(qtbot):
    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    all_op_ids = set()
    for i in range(library.tree.topLevelItemCount()):
        category_item = library.tree.topLevelItem(i)
        assert category_item.childCount() > 0
        for j in range(category_item.childCount()):
            child = category_item.child(j)
            all_op_ids.add(child.data(0, Qt.ItemDataRole.UserRole))
            assert child.toolTip(0)  # her operatörün bir açıklaması olmalı

    assert all_op_ids == set(registry.ids())


def test_checking_item_emits_operator_checked(qtbot):
    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    item = library._items_by_op_id["segment.threshold"]
    received = []
    library.operator_checked.connect(received.append)
    library.operator_unchecked.connect(lambda op_id: received.append(f"unchecked:{op_id}"))

    item.setCheckState(0, Qt.CheckState.Checked)

    assert received == ["segment.threshold"]


def test_unchecking_item_emits_operator_unchecked(qtbot):
    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    item = library._items_by_op_id["segment.threshold"]
    item.setCheckState(0, Qt.CheckState.Checked)

    received = []
    library.operator_unchecked.connect(received.append)
    item.setCheckState(0, Qt.CheckState.Unchecked)

    assert received == ["segment.threshold"]


def test_set_checked_does_not_emit_signal(qtbot):
    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    received = []
    library.operator_checked.connect(received.append)
    library.set_checked("segment.threshold", True)

    assert received == []
    assert library._items_by_op_id["segment.threshold"].checkState(0) == Qt.CheckState.Checked


def test_custom_filter_categorized_under_filtreler_with_saved_name(_isolated_custom_filter_dir):
    custom_filters.save_custom_filter(
        custom_filters.CustomFilterDef(name="Benim Filtrem", code="def apply(image):\n    return image\n")
    )
    op_id = custom_filters.op_id_for("Benim Filtrem")

    assert category_for(op_id) == "Filtreler"
    assert label_for(op_id) == "Benim Filtrem"
    assert description_for(op_id) != op_id


def test_custom_filter_button_emits_editor_requested_signal(qtbot):
    from PySide6.QtWidgets import QPushButton

    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    received = []
    library.custom_filter_editor_requested.connect(lambda: received.append(1))
    button = next(b for b in library.findChildren(QPushButton) if "Özel Filtre" in b.text())
    button.click()

    assert received == [1]


def test_search_hides_non_matching_operators(qtbot):
    library = OperatorLibrary(registry)
    qtbot.addWidget(library)

    library.search_box.setText("threshold")

    visible_op_ids = set()
    for i in range(library.tree.topLevelItemCount()):
        category_item = library.tree.topLevelItem(i)
        if category_item.isHidden():
            continue
        for j in range(category_item.childCount()):
            child = category_item.child(j)
            if not child.isHidden():
                visible_op_ids.add(child.data(0, Qt.ItemDataRole.UserRole))

    assert visible_op_ids == {"segment.threshold"}
