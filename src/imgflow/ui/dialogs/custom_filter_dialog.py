"""Kullanıcının OpenCV kodu yazarak kendi filtresini eklediği/düzenlediği/sildiği pencere.

`frame_provider` (mevcut kamera/pipeline karesi, yoksa None döner) ile test edilebilir bir
görüntü sağlanır; kamera açık değilse kullanıcı diskten bir test görüntüsü seçebilir.
Kaydedilen filtre `core/custom_filters.py` üzerinden hem diske hem de operatör registry'sine
yazılır ve `filters_changed` sinyaliyle operatör kütüphanesi panelinin yenilenmesi sağlanır.

Kod doğrudan exec() ile çalıştırıldığı için (bkz. `core/custom_filters.py` docstring'i) bu
NOT kullanıcıya pencerede açıkça gösterilir — sahte bir güvenlik hissi yaratılmaz.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.custom_filters import (
    CustomFilterDef,
    CustomFilterError,
    DEFAULT_TEMPLATE,
    delete_custom_filter,
    list_custom_filters,
    op_id_for,
    preview,
    save_custom_filter,
)
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import ImageView

_DISCLAIMER = (
    "Bu kod doğrudan çalıştırılır (exec) — yalnızca kendi yazdığınız/güvendiğiniz kodu "
    "kullanın. Bir sanal alan (sandbox) değildir."
)


class CustomFilterDialog(QDialog):
    filters_changed = Signal()

    def __init__(
        self,
        frame_provider: Callable[[], np.ndarray | None],
        is_op_in_use: Callable[[str], bool] | None = None,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Özel Filtre Ekle/Düzenle")
        self.setModal(False)
        self.resize(760, 560)
        self._frame_provider = frame_provider
        self._is_op_in_use = is_op_in_use if is_op_in_use is not None else (lambda _op_id: False)
        self._test_image: np.ndarray | None = None

        self._list = QListWidget()
        self._list.currentItemChanged.connect(self._on_selection_changed)
        new_button = QPushButton("Yeni Filtre")
        new_button.clicked.connect(self._on_new)
        self._delete_button = QPushButton("Sil")
        self._delete_button.clicked.connect(self._on_delete)
        self._delete_button.setEnabled(False)

        left_layout = QVBoxLayout()
        left_layout.addWidget(QLabel("Kayıtlı Özel Filtreler"))
        left_layout.addWidget(self._list, 1)
        left_buttons = QHBoxLayout()
        left_buttons.addWidget(new_button)
        left_buttons.addWidget(self._delete_button)
        left_layout.addLayout(left_buttons)
        left_widget = QWidget()
        left_widget.setLayout(left_layout)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Filtre adı (ör. Özel Kenar Vurgusu)")

        self._code_edit = QPlainTextEdit()
        self._code_edit.setFont(QFont("Consolas", 10))
        self._code_edit.setPlainText(DEFAULT_TEMPLATE)

        test_button = QPushButton("Test Et")
        test_button.clicked.connect(self._on_test)
        load_image_button = QPushButton("Test Görüntüsü Yükle...")
        load_image_button.clicked.connect(self._on_load_test_image)
        self._save_button = QPushButton("Kaydet")
        self._save_button.clicked.connect(self._on_save)

        test_row = QHBoxLayout()
        test_row.addWidget(load_image_button)
        test_row.addWidget(test_button)
        test_row.addStretch(1)
        test_row.addWidget(self._save_button)

        self._preview = ImageView()
        self._error_label = QLabel("")
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c0392b;")
        self._status_label = QLabel("")

        right_layout = QVBoxLayout()
        right_layout.addWidget(QLabel(_DISCLAIMER))
        right_layout.addWidget(self._name_edit)
        right_layout.addWidget(self._code_edit, 2)
        right_layout.addLayout(test_row)
        right_layout.addWidget(self._preview, 1)
        right_layout.addWidget(self._error_label)
        right_layout.addWidget(self._status_label)
        right_widget = QWidget()
        right_widget.setLayout(right_layout)

        splitter = QSplitter()
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

        self._refresh_list()

    # -- yardımcılar --------------------------------------------------------

    def _refresh_list(self, select_name: str | None = None) -> None:
        self._list.blockSignals(True)
        self._list.clear()
        for defn in list_custom_filters():
            item = QListWidgetItem(defn.name)
            self._list.addItem(item)
            if select_name is not None and defn.name == select_name:
                self._list.setCurrentItem(item)
        self._list.blockSignals(False)
        self._delete_button.setEnabled(self._list.currentItem() is not None)

    def _current_test_image(self) -> np.ndarray | None:
        if self._test_image is not None:
            return self._test_image
        return self._frame_provider()

    # -- olaylar --------------------------------------------------------

    def _on_selection_changed(self, current: QListWidgetItem | None, _previous) -> None:
        self._delete_button.setEnabled(current is not None)
        if current is None:
            return
        defn = next((d for d in list_custom_filters() if d.name == current.text()), None)
        if defn is None:
            return
        self._name_edit.setText(defn.name)
        self._code_edit.setPlainText(defn.code)
        self._error_label.setText("")
        self._status_label.setText("")

    def _on_new(self) -> None:
        self._list.setCurrentItem(None)
        self._name_edit.clear()
        self._code_edit.setPlainText(DEFAULT_TEMPLATE)
        self._error_label.setText("")
        self._status_label.setText("")

    def _on_load_test_image(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Test Görüntüsü Seç", "", "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        try:
            self._test_image = load_image(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Görüntü Yüklenemedi", str(exc))
            return
        self._preview.set_image(self._test_image)

    def _on_test(self) -> None:
        image = self._current_test_image()
        if image is None:
            self._error_label.setText(
                "Test edilecek bir görüntü yok — kamera açın ya da 'Test Görüntüsü Yükle...' ile bir dosya seçin."
            )
            return
        name = self._name_edit.text().strip() or "Adsız Filtre"
        try:
            result = preview(CustomFilterDef(name=name, code=self._code_edit.toPlainText()), image)
        except CustomFilterError as exc:
            self._error_label.setText(str(exc))
            return
        self._error_label.setText("")
        self._preview.set_image(result)

    def _on_save(self) -> None:
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Ad Gerekli", "Kaydetmek için bir filtre adı girin.")
            return
        defn = CustomFilterDef(name=name, code=self._code_edit.toPlainText())
        try:
            save_custom_filter(defn)
        except CustomFilterError as exc:
            self._error_label.setText(str(exc))
            return
        self._error_label.setText("")
        self._status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_list(select_name=name)
        self.filters_changed.emit()

    def _on_delete(self) -> None:
        item = self._list.currentItem()
        if item is None:
            return
        name = item.text()
        defn = next((d for d in list_custom_filters() if d.name == name), None)
        if defn is None:
            return
        if self._is_op_in_use(op_id_for(name)):
            QMessageBox.warning(
                self,
                "Kullanımda",
                f"'{name}' filtresi şu an pipeline'da kullanılıyor. Silmeden önce pipeline'dan kaldırın.",
            )
            return
        reply = QMessageBox.question(self, "Filtreyi Sil", f"'{name}' filtresi kalıcı olarak silinsin mi?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        delete_custom_filter(name)
        self._on_new()
        self._refresh_list()
        self.filters_changed.emit()
