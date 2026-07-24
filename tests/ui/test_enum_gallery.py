import numpy as np

from imgflow.ui.widgets.enum_gallery import EnumGallery


def test_show_choices_creates_one_button_per_choice(qtbot):
    gallery = EnumGallery()
    qtbot.addWidget(gallery)

    def render(choice: str):
        return np.zeros((4, 4), dtype=np.uint8)

    gallery.show_choices("mode", ["BGR2GRAY", "BGR2HSV"], "BGR2GRAY", render)

    buttons = [gallery._row_layout.itemAt(i).widget() for i in range(gallery._row_layout.count())]
    buttons = [b for b in buttons if b is not None and hasattr(b, "text")]
    assert [b.text() for b in buttons] == ["BGR2GRAY", "BGR2HSV"]
    assert buttons[0].isChecked() is True
    assert buttons[1].isChecked() is False


def test_clicking_a_choice_emits_choice_selected(qtbot):
    gallery = EnumGallery()
    qtbot.addWidget(gallery)

    def render(choice: str):
        return np.zeros((4, 4), dtype=np.uint8)

    gallery.show_choices("mode", ["BINARY", "OTSU"], "BINARY", render)
    buttons = [gallery._row_layout.itemAt(i).widget() for i in range(gallery._row_layout.count())]
    otsu_button = next(b for b in buttons if b is not None and hasattr(b, "text") and b.text() == "OTSU")

    received = []
    gallery.choice_selected.connect(lambda param, choice: received.append((param, choice)))
    otsu_button.click()

    assert received == [("mode", "OTSU")]


def test_show_choices_handles_none_render_result(qtbot):
    gallery = EnumGallery()
    qtbot.addWidget(gallery)

    gallery.show_choices("mode", ["X"], "X", lambda choice: None)

    buttons = [gallery._row_layout.itemAt(i).widget() for i in range(gallery._row_layout.count())]
    buttons = [b for b in buttons if b is not None and hasattr(b, "text")]
    assert len(buttons) == 1


def test_clear_removes_previous_buttons(qtbot):
    gallery = EnumGallery()
    qtbot.addWidget(gallery)
    gallery.show_choices("mode", ["A", "B"], "A", lambda choice: None)

    gallery.clear()

    buttons = [gallery._row_layout.itemAt(i).widget() for i in range(gallery._row_layout.count())]
    buttons = [b for b in buttons if b is not None and hasattr(b, "text")]
    assert buttons == []
