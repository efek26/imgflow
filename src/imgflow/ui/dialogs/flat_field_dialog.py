"""Aydınlatma Referansı Kaydet — bağımsız non-modal QDialog.

Kullanıcı BOŞ/düz aydınlatılmış bir kare (ör. üzerinde ürün olmayan boş bant) yükler/
sürükler, isimle kaydeder; pipeline'daki `correction.flat_field` operatörü bu referansı
`reference_name` parametresiyle kullanır. `ShapeMatchingDialog`'daki isimli-profil
deseninin (yükle → isim ver → Kaydet; Yükle/Sil ile yönetim) sadeleştirilmiş hali — burada
ROI/eğitim adımı yok, referans doğrudan yüklenen görüntünün kendisi.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from imgflow.io_utils import flatfield_store
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import ImageView

_GUIDE_TEXT = (
    "BOŞ/düz aydınlatılmış bir kare yükleyin ya da sürükleyip bırakın (ör. üzerinde ürün "
    "olmayan boş bant) → isim verip 'Kaydet'. Pipeline'da 'Aydınlatma Düzeltme' "
    "kategorisindeki operatörde bu isim 'Aydınlatma Referansı' açılır listesinden "
    "seçilebilir. Kayıtlı referansları aşağıdaki listeden Sil ile yönetebilirsiniz."
)


class FlatFieldDialog(QDialog):
    references_changed = Signal(str)
    """Kayıtlı referans KÜMESİ değiştiğinde (kaydet/sil) yayınlanır — ana pencere,
    `correction.flat_field` düğümü seçiliyse parametre panelindeki 'Aydınlatma Referansı'
    açılır listesini bu sinyalle canlı olarak yeniler (bkz. `_on_shape_models_changed`'in
    ONNX/şekil eşleştirmedeki aynı deseni). Bir KAYDETME'den geliyorsa parametre az önce
    kaydedilen ismi taşır (silmede boş string) -- gerçek kullanıcı raporu: referansı
    kaydetmek pipeline adımının 'reference_name' parametresini OTOMATİK seçmiyordu, kullanıcı
    referansı kaydettiğini/yüklediğini düşünüp filtreyi uyguluyor ama alan hâlâ boş kaldığı
    için "'reference_name' parametresi boş olamaz" hatası alıyordu. Ana pencere bu ismi,
    SADECE seçili adımın `reference_name` alanı hâlâ BOŞSA otomatik doldurmak için kullanır
    (doluysa kullanıcının üzerinde çalıştığı başka bir referansı sessizce değiştirmez)."""

    def __init__(self, directory: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aydınlatma Referansı Kaydet")
        self.setModal(False)
        self._directory = directory if directory is not None else flatfield_store.FLATFIELD_DIR
        self._image: np.ndarray | None = None

        self._view = ImageView()
        self._view.setAcceptDrops(True)
        self._view.image_file_dropped.connect(self._on_file_dropped)

        load_button = QPushButton("Görüntü Yükle...")
        load_button.clicked.connect(self._on_load)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Referans adı (ör. hat1_bos_bant)")
        self._save_button = QPushButton("Kaydet")
        self._save_button.setToolTip("Yüklenen görüntüyü yukarıdaki isimle diske kaydeder.")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save)
        self._status_label = QLabel("")

        self._list_combo = QComboBox()
        self._list_combo.setToolTip("Daha önce kaydedilmiş aydınlatma referansları.")
        delete_button = QPushButton("Sil")
        delete_button.setToolTip("Seçili referansı kalıcı olarak siler.")
        delete_button.clicked.connect(self._on_delete)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        save_row = QHBoxLayout()
        save_row.addWidget(self._name_edit, 1)
        save_row.addWidget(self._save_button)

        manage_row = QHBoxLayout()
        manage_row.addWidget(self._list_combo, 1)
        manage_row.addWidget(delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addWidget(self._view, 1)
        layout.addWidget(load_button)
        layout.addLayout(save_row)
        layout.addWidget(self._status_label)
        layout.addWidget(QLabel("Kayıtlı Referanslar:"))
        layout.addLayout(manage_row)

        self._refresh_list()

    def _on_load(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self, "Referans Görüntü Seç", "", "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not path:
            return
        self._load_path(path)

    def _on_file_dropped(self, path: str) -> None:
        self._load_path(path)

    def _load_path(self, path: str) -> None:
        try:
            image = load_image(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Görüntü Yüklenemedi", str(exc))
            return
        self._image = image
        self._view.set_image(image)
        self._save_button.setEnabled(True)
        self._status_label.setText("")

    def _on_save(self) -> None:
        if self._image is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Ad Gerekli", "Kaydetmek için bir referans adı girin.")
            return
        flatfield_store.save_reference(name, self._image, directory=self._directory)
        self._status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_list()
        self.references_changed.emit(name)

    def _refresh_list(self) -> None:
        current = self._list_combo.currentText()
        self._list_combo.clear()
        names = flatfield_store.list_references(self._directory)
        self._list_combo.addItems(names)
        if current in names:
            self._list_combo.setCurrentText(current)

    def _on_delete(self) -> None:
        name = self._list_combo.currentText().strip()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Referansı Sil", f"'{name}' adlı referans kalıcı olarak silinsin mi?"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        flatfield_store.delete_reference(name, directory=self._directory)
        self._status_label.setText(f"Silindi: '{name}'")
        self._refresh_list()
        self.references_changed.emit("")
