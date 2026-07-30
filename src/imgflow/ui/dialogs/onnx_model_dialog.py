"""ONNX Model Kaydet — bağımsız non-modal QDialog.

Kullanıcı imgflow DIŞINDA (ör. Ultralytics YOLO ile) eğitilmiş bir `.onnx` dosyasını seçer,
model türünü + giriş boyutunu + sınıf isimlerini belirtip isimle kaydeder
(`io_utils/onnx_model_store.py`) — pipeline'daki `ml.onnx_detect` operatörü bu referansı
`model_names` parametresiyle kullanır. `ShapeMatchingDialog`/`FlatFieldDialog`'daki isimli-
profil deseninin (seç → meta bilgi gir → Kaydet; Sil ile yönetim) aynısı.

**Model Türü** seçeneklerinden şu an SADECE "YOLO" çalıştırılabilir; "Sınıflandırma"/
"Segmentasyon" de kaydedilebilir (gelecek fazlar için yer tutucu, bkz.
`io_utils/onnx_model_store.py` modül docstring'i) ama `ml.onnx_detect` operatörü bu
türdeki bir modelle çalıştırılırsa net bir "henüz desteklenmiyor" hatası verir.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from imgflow.core.onnx_detection import inspect_onnx_model
from imgflow.io_utils import onnx_model_store

_GUIDE_TEXT = (
    "imgflow DIŞINDA (ör. Ultralytics YOLO ile) eğitilmiş bir '.onnx' dosyası seçin — sınıf "
    "isimleri ve giriş boyutu, modelin kendi metadata'sından OTOMATİK doldurulur (okunamazsa "
    "elle girersiniz) — model türünü seçip 'Kaydet'. Pipeline'da 'Yapay Zeka (ONNX)' "
    "kategorisindeki operatörde bu isim 'Model(ler)' açılır listesinden seçilebilir."
)
_TASK_TYPE_LABELS = {
    onnx_model_store.TASK_TYPE_YOLO: "YOLO",
    onnx_model_store.TASK_TYPE_CLASSIFICATION: "Sınıflandırma / Anomali",
    onnx_model_store.TASK_TYPE_SEGMENTATION: "Segmentasyon",
}
_TASK_TYPE_BY_LABEL = {label: task_type for task_type, label in _TASK_TYPE_LABELS.items()}


class OnnxModelDialog(QDialog):
    models_changed = Signal()
    """Kayıtlı model KÜMESİ değiştiğinde (kaydet/sil) yayınlanır — `_on_shape_models_changed`/
    `_on_flat_field_references_changed` ile AYNI desen: ana pencere, `ml.onnx_detect` düğümü
    seçiliyse parametre panelindeki 'Model(ler)' açılır listesini bu sinyalle canlı yeniler."""

    def __init__(self, directory: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("ONNX Model Kaydet")
        self.setModal(False)
        self._directory = directory if directory is not None else onnx_model_store.ONNX_MODEL_DIR
        self._selected_onnx_path: Path | None = None

        self._path_label = QLabel("(henüz seçilmedi)")
        self._path_label.setWordWrap(True)
        choose_button = QPushButton(".onnx Dosyası Seç...")
        choose_button.clicked.connect(self._on_choose_file)

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Model adı (ör. kusur_dedektoru)")

        self._task_type_combo = QComboBox()
        self._task_type_combo.addItems(list(_TASK_TYPE_LABELS.values()))

        self._input_size_spin = QSpinBox()
        self._input_size_spin.setRange(32, 4096)
        self._input_size_spin.setSingleStep(32)
        self._input_size_spin.setValue(640)
        self._input_size_spin.setToolTip(
            "Modelin eğitildiği kare giriş boyutu (px) — YOLO ihracatlarında genelde 640."
        )

        self._class_labels_edit = QLineEdit()
        self._class_labels_edit.setPlaceholderText("Sınıf isimleri, virgülle (ör. kusur, saglam)")

        self._save_button = QPushButton("Kaydet")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save)
        self._status_label = QLabel("")

        self._list_combo = QComboBox()
        self._list_combo.setToolTip("Daha önce kaydedilmiş ONNX modelleri.")
        delete_button = QPushButton("Sil")
        delete_button.setToolTip("Seçili modeli kalıcı olarak siler.")
        delete_button.clicked.connect(self._on_delete)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        file_row = QHBoxLayout()
        file_row.addWidget(self._path_label, 1)
        file_row.addWidget(choose_button)

        form = QFormLayout()
        form.addRow("Model Adı:", self._name_edit)
        form.addRow("Model Türü:", self._task_type_combo)
        form.addRow("Giriş Boyutu:", self._input_size_spin)
        form.addRow("Sınıf İsimleri:", self._class_labels_edit)

        manage_row = QHBoxLayout()
        manage_row.addWidget(self._list_combo, 1)
        manage_row.addWidget(delete_button)

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addLayout(file_row)
        layout.addLayout(form)
        layout.addWidget(self._save_button)
        layout.addWidget(self._status_label)
        layout.addWidget(QLabel("Kayıtlı Modeller:"))
        layout.addLayout(manage_row)

        self._refresh_list()

    def _on_choose_file(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(self, "ONNX Model Seç", "", "ONNX Modelleri (*.onnx)")
        if not path:
            return
        self._selected_onnx_path = Path(path)
        self._path_label.setText(path)
        self._save_button.setEnabled(True)
        self._autofill_from_model(self._selected_onnx_path)

    def _autofill_from_model(self, path: Path) -> None:
        """Sınıf isimlerini ve giriş boyutunu modelin KENDİ metadata'sından doldurur — böylece
        kullanıcı virgüllü listeyi elle yeniden yazmaz ve SIRA uyuşmazlığından doğan sessiz
        yanlış etiketleme (gerçek kullanıcı şikayeti: "sınıflandırmalar bozuk/hatalı gibi
        geliyor") olmaz. Okunamayan/bozuk dosyada (bkz. `inspect_onnx_model` ASLA fırlatmaz)
        alanlar DEĞİŞMEDEN bırakılır ve kullanıcı elle girmeye yönlendirilir."""
        info = inspect_onnx_model(path)
        filled = []
        if info["class_labels"]:
            self._class_labels_edit.setText(", ".join(info["class_labels"]))
            filled.append("sınıf isimleri")
        if info["input_size"]:
            self._input_size_spin.setValue(info["input_size"])
            filled.append("giriş boyutu")

        if filled:
            self._status_label.setText(f"Modelden otomatik okundu: {' ve '.join(filled)}.")
        else:
            self._status_label.setText(
                "Model metadata'sı okunamadı — sınıf isimlerini ve giriş boyutunu elle girin."
            )

    def _on_save(self) -> None:
        if self._selected_onnx_path is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Ad Gerekli", "Kaydetmek için bir model adı girin.")
            return
        class_labels = [c.strip() for c in self._class_labels_edit.text().split(",") if c.strip()]
        if not class_labels:
            QMessageBox.warning(self, "Sınıf İsmi Gerekli", "En az bir sınıf ismi girin.")
            return
        task_type = _TASK_TYPE_BY_LABEL[self._task_type_combo.currentText()]

        onnx_model_store.save_model(
            name,
            self._selected_onnx_path,
            task_type,
            self._input_size_spin.value(),
            class_labels,
            directory=self._directory,
        )
        self._status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_list()
        self.models_changed.emit()

    def _refresh_list(self) -> None:
        current = self._list_combo.currentText()
        self._list_combo.clear()
        names = onnx_model_store.list_models(self._directory)
        self._list_combo.addItems(names)
        if current in names:
            self._list_combo.setCurrentText(current)

    def _on_delete(self) -> None:
        name = self._list_combo.currentText().strip()
        if not name:
            return
        reply = QMessageBox.question(self, "Modeli Sil", f"'{name}' adlı model kalıcı olarak silinsin mi?")
        if reply != QMessageBox.StandardButton.Yes:
            return
        onnx_model_store.delete_model(name, directory=self._directory)
        self._status_label.setText(f"Silindi: '{name}'")
        self._refresh_list()
        self.models_changed.emit()
