from PySide6.QtCore import QPoint, QRect

from imgflow.ui.widgets import _display_transform as dt


def test_display_rect_centers_and_scales_image():
    rect = dt.display_rect((100, 50), (200, 200))
    assert (rect.width(), rect.height()) == (200, 100)
    assert rect.y() == 50
    assert rect.x() == 0


def test_display_rect_empty_for_zero_image_size():
    assert dt.display_rect((0, 0), (200, 200)) == QRect()


def test_widget_point_to_image_round_trips_with_image_point_to_widget():
    image_size = (100, 100)
    widget_size = (200, 200)  # scale = 2

    widget_point = dt.image_point_to_widget((30, 40), image_size, widget_size)
    back = dt.widget_point_to_image(widget_point, image_size, widget_size)

    assert back == (30, 40)


def test_widget_point_to_image_clamps_to_bounds():
    x, y = dt.widget_point_to_image(QPoint(-50, -50), (100, 100), (200, 200))
    assert (x, y) == (0, 0)


def test_widget_rect_to_image_and_back():
    image_size = (100, 100)
    widget_size = (200, 200)
    image_rect = (10, 10, 20, 30)

    widget_rect = dt.image_rect_to_widget(image_rect, image_size, widget_size)
    back = dt.widget_rect_to_image(widget_rect, image_size, widget_size)

    assert back == image_rect


def test_widget_circle_to_image_and_back():
    image_size = (100, 100)
    widget_size = (200, 200)

    widget_circle = dt.image_circle_to_widget(30, 30, 10, image_size, widget_size)
    assert widget_circle is not None
    center, radius = widget_circle
    back = dt.widget_circle_to_image(center, radius, image_size, widget_size)

    assert back == (30, 30, 10)


def test_image_circle_to_widget_none_when_display_empty():
    assert dt.image_circle_to_widget(1, 1, 1, (0, 0), (200, 200)) is None
