"""Seçili node'un ENUM parametresi için her seçeneğin küçük önizlemesini yan yana gösteren galeri.

Kullanıcı bir dropdown'dan metin okumak yerine sonucu görüp tıklayarak seçebilir.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QScrollArea, QVBoxLayout, QWidget

from imgflow.ui.widgets.image_view import normalize_to_uint8, numpy_to_qimage

_THUMB_SIZE = 88


class EnumGallery(QWidget):
    choice_selected = Signal(str, str)  # (param_name, choice)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self._row = QWidget()
        self._row_layout = QHBoxLayout(self._row)
        self._row_layout.setContentsMargins(4, 4, 4, 4)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self._row)
        scroll.setFixedHeight(_THUMB_SIZE + 50)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        outer_layout.addWidget(scroll)

    def clear(self) -> None:
        while self._row_layout.count():
            item = self._row_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def show_choices(
        self,
        param_name: str,
        choices: list[str],
        current: str,
        render: Callable[[str], "np.ndarray | None"],
    ) -> None:
        self.clear()
        for choice in choices:
            button = QPushButton(choice)
            button.setCheckable(True)
            button.setChecked(choice == current)
            button.setIconSize(QSize(_THUMB_SIZE, _THUMB_SIZE))
            button.setToolTip(choice)

            image = render(choice)
            if image is not None:
                pixmap = self._to_thumbnail(image)
                button.setIcon(QIcon(pixmap))

            button.clicked.connect(lambda checked=False, c=choice: self.choice_selected.emit(param_name, c))
            self._row_layout.addWidget(button)
        self._row_layout.addStretch(1)

    def _to_thumbnail(self, image: np.ndarray) -> QPixmap:
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        return QPixmap.fromImage(qimage).scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
