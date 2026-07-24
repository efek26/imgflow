"""Şekil Eşleştirme (Model Öğret) — bağımsız non-modal QDialog.

Referans bir görüntü yüklenir, `RoiCanvas` üzerinde (dikdörtgen) model bölgesi seçilir,
"Modeli Eğit" ile `core/shape_matching.train_shape_model()` çağrılır, ve model isimle
kaydedilir (`io_utils/shape_model_store.py`) — daha sonra pipeline'daki "Geometrik
Eşleştirme" operatöründen (`geom.shape_match`) `model_names` parametresiyle (birden fazla model
aynı adımda seçilebilir) kullanılabilir.

Lens/Yükseklik-Ölçek kalibrasyon dialoglarıyla (bkz. `height_scale_calibration_dialog.py`)
AYNI isimli-profil deseni: eğit → isim ver → Kaydet; kayıtlı modeller açılır listeden
Yükle/Sil/Yeniden Adlandır. Pipeline'ın kendisinden TAMAMEN bağımsızdır — model burada bir
kere eğitilir, operatör onu sadece isimle referans alır (ikinci bir zorunlu girdiye ihtiyaç
duymadan).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from imgflow.core.roi import RoiRect
from imgflow.core.shape_matching import ShapeMatchingError, ShapeModel, train_shape_model
from imgflow.io_utils import shape_model_store
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import normalize_to_uint8, numpy_to_qimage
from imgflow.ui.widgets.roi_canvas import RoiCanvas

_GUIDE_TEXT = (
    "Referans Görüntüler Yükle (birden fazla seçilebilir) → galeriden birini seçin → model "
    "bölgesini (dikdörtgen) fare ile çiz/taşı/boyutlandır → 'Modeli Eğit' → isim verip "
    "'Kaydet'. Pipeline'da 'Geometrik Eşleştirme' kategorisindeki operatörde bu isim 'Model "
    "Adı' açılır listesinden seçilebilir. Kayıtlı modelleri aşağıdaki listeden Yükle/Sil/"
    "Yeniden Adlandır ile yönetebilirsiniz."
)
_DEFAULT_ROI = (40, 40, 120, 120)
_THUMB_SIZE = 96
_REFERENCE_INDEX_ROLE = Qt.ItemDataRole.UserRole
_NUM_LEVELS_HELP = (
    "Piramit Seviyesi: aramanın kaç çözünürlük kademesinde (tam boy → yarı boy → çeyrek boy ...) "
    "yapılacağı. Önce EN KABA (en küçük) seviyede hızlıca birkaç aday konum/açı bulunur, sonra "
    "SADECE bu adaylar daha ince seviyelerde iyileştirilir — bu yüzden yüksek değer aramayı asıl "
    "yavaşlatmaz, aksine daha kaba bir ön-elemeyle genelde hızlandırır. Çok küçük/detaysız "
    "modellerde (küçük ROI) 1-2 seviye yeterlidir; büyük/detaylı modellerde 3-4 önerilir."
)
_MIN_GRADIENT_HELP = (
    "Min. Gradyan Oranı: bir pikselin model için 'kenar noktası' sayılması için gereken gradyan "
    "(parlaklık değişimi) büyüklüğü eşiği — ROI içindeki EN GÜÇLÜ kenara ORANLA (mutlak bir "
    "piksel değeri değil). Düşürürseniz daha SOLUK/zayıf kenarlar da modele dahil edilir (daha "
    "fazla nokta, gürültüye karşı daha az sağlam); yükseltirseniz sadece EN BELİRGİN kenarlar "
    "kullanılır (daha az ama daha güvenilir nokta). ROI'de kenar bulunamazsa eğitim hata verir — "
    "bu durumda bu değeri düşürmeyi deneyin."
)


class ShapeMatchingDialog(QDialog):
    model_trained = Signal(object)  # ShapeModel
    models_changed = Signal()
    """Kayıtlı model KÜMESİ değiştiğinde (kaydet/sil/yeniden adlandır) yayınlanır — ana
    pencere, `geom.shape_match` düğümü seçiliyse parametre panelindeki 'Model Adı' açılır
    listesini bu sinyalle canlı olarak yeniler."""

    def __init__(self, model_dir: Path | None = None, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Şekil Eşleştirme (Model Öğret)")
        self.setModal(False)
        self._model_dir = model_dir if model_dir is not None else shape_model_store.SHAPE_MODEL_DIR
        self._reference_image: np.ndarray | None = None
        self._reference_images: list[np.ndarray] = []
        """`_reference_list` galerisindeki her thumbnail'e karşılık gelen tam-çözünürlük
        görüntü, sırasıyla — kullanıcı "daha fazla fotoğraf yüklenebilmeli, daha yüksek
        doğruluk payımız olsun" istedi: birden fazla referans fotoğrafı aynı anda yüklenip
        galeride bir arada tutulabilir, aralarında tıklayarak en temiz/net olanı seçip o
        pozdan model eğitilebilir (fotoğraflar BİRLEŞTİRİLMEZ — kullanıcı bilinçli olarak bu
        basit galeri yaklaşımını, kenar noktalarını birleştiren daha karmaşık alternatife
        tercih etti)."""
        self._reference_names: list[str] = []
        self._active_reference_index: int | None = None
        """Galerideki HANGİ görüntünün şu an eğitim referansı (dolayısıyla modelin 'açı=0'
        baz aldığı poz) olduğu, listede '★' öneki + `_active_reference_label` ile AÇIKÇA
        gösterilir — kullanıcı isteği: "model eğitme kısmında açının doğru alınması için bir
        tane referans seçelim". Birden fazla foto arasında hangisinin eğitime gireceği belirsiz
        kalırsa, farklı fotoğraflardaki küçük poz farkları modelin açı referansını kaydırabilir."""
        self._model: ShapeModel | None = None
        self._roi = _DEFAULT_ROI

        self._canvas = RoiCanvas()
        self._canvas.set_shape("RECT")
        self._canvas.set_editing_enabled(True)
        self._canvas.roi_changed.connect(self._on_roi_changed)
        self._canvas.setAcceptDrops(True)
        self._canvas.image_file_dropped.connect(self._on_reference_file_dropped)

        load_button = QPushButton("Referans Görüntüler Yükle...")
        load_button.setToolTip("Birden fazla görüntü seçebilirsiniz — hepsi aşağıdaki galeriye eklenir.")
        load_button.clicked.connect(self._on_load_references)

        self._reference_list = QListWidget()
        self._reference_list.setViewMode(QListWidget.ViewMode.IconMode)
        self._reference_list.setIconSize(QPixmap(_THUMB_SIZE, _THUMB_SIZE).size())
        self._reference_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._reference_list.setFixedHeight(_THUMB_SIZE + 48)
        self._reference_list.setToolTip("Bir referansa tıklayarak canvas'ta aktif hale getirin.")
        self._reference_list.itemClicked.connect(self._on_reference_gallery_item_clicked)
        self._active_reference_label = QLabel("Aktif referans: (henüz yok)")

        self._num_levels_spin = QSpinBox()
        self._num_levels_spin.setRange(1, 6)
        self._num_levels_spin.setValue(3)
        self._num_levels_spin.setToolTip(_NUM_LEVELS_HELP)
        num_levels_label = QLabel("Piramit Seviyesi:")
        num_levels_label.setToolTip(_NUM_LEVELS_HELP)

        self._min_gradient_spin = QDoubleSpinBox()
        self._min_gradient_spin.setRange(0.01, 0.95)
        self._min_gradient_spin.setSingleStep(0.05)
        self._min_gradient_spin.setValue(0.2)
        self._min_gradient_spin.setToolTip(_MIN_GRADIENT_HELP)
        min_gradient_label = QLabel("Min. Gradyan Oranı:")
        min_gradient_label.setToolTip(_MIN_GRADIENT_HELP)

        train_button = QPushButton("Modeli Eğit")
        train_button.setToolTip(
            "Seçili dikdörtgen bölgedeki kenarlardan bir şekil modeli çıkarır. Modelin "
            "kaydedilmesi için önce bu adımın başarıyla tamamlanması gerekir."
        )
        train_button.clicked.connect(self._on_train)
        self._train_status_label = QLabel("")

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Model adı (ör. kapak_logo)")
        self._save_button = QPushButton("Kaydet")
        self._save_button.setToolTip("Son eğitilen/yüklenen modeli yukarıdaki isimle diske kaydeder.")
        self._save_button.setEnabled(False)
        self._save_button.clicked.connect(self._on_save)
        self._save_status_label = QLabel("")

        self._load_combo = QComboBox()
        self._load_combo.setToolTip("Daha önce kaydedilmiş şekil modelleri.")
        load_model_button = QPushButton("Yükle")
        load_model_button.setToolTip("Seçili modeli belleğe yükler (pipeline'daki operatörden bağımsız).")
        load_model_button.clicked.connect(self._on_load_model)
        rename_button = QPushButton("Yeniden Adlandır...")
        rename_button.setToolTip("Seçili modelin adını değiştirir.")
        rename_button.clicked.connect(self._on_rename_model)
        delete_button = QPushButton("Sil")
        delete_button.setToolTip("Seçili modeli kalıcı olarak siler.")
        delete_button.clicked.connect(self._on_delete_model)

        params_row = QHBoxLayout()
        params_row.addWidget(num_levels_label)
        params_row.addWidget(self._num_levels_spin)
        params_row.addWidget(min_gradient_label)
        params_row.addWidget(self._min_gradient_spin)

        save_row = QHBoxLayout()
        save_row.addWidget(self._name_edit, 1)
        save_row.addWidget(self._save_button)

        manage_row = QHBoxLayout()
        manage_row.addWidget(self._load_combo, 1)
        manage_row.addWidget(load_model_button)
        manage_row.addWidget(rename_button)
        manage_row.addWidget(delete_button)

        guide_label = QLabel(_GUIDE_TEXT)
        guide_label.setWordWrap(True)
        guide_label.setStyleSheet("color: #666; font-style: italic;")

        layout = QVBoxLayout(self)
        layout.addWidget(guide_label)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(load_button)
        layout.addWidget(QLabel("Referans Galerisi:"))
        layout.addWidget(self._reference_list)
        layout.addWidget(self._active_reference_label)
        layout.addLayout(params_row)
        layout.addWidget(train_button)
        layout.addWidget(self._train_status_label)
        layout.addLayout(save_row)
        layout.addWidget(self._save_status_label)
        layout.addWidget(QLabel("Kayıtlı Modeller:"))
        layout.addLayout(manage_row)

        self._refresh_model_combo()

    @property
    def model(self) -> ShapeModel | None:
        return self._model

    def _on_roi_changed(self, x: int, y: int, w: int, h: int) -> None:
        self._roi = (x, y, w, h)

    def _on_load_references(self) -> None:
        paths, _filter = QFileDialog.getOpenFileNames(
            self, "Referans Görüntüler Seç", "", "Görüntüler (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )
        if not paths:
            return
        last_index: int | None = None
        for path in paths:
            try:
                image = load_image(path)
            except FileNotFoundError as exc:
                QMessageBox.critical(self, "Görüntü Yüklenemedi", f"{path}: {exc}")
                continue
            last_index = self._add_reference_to_gallery(path, image)
        if last_index is not None:
            self._activate_reference(last_index)

    def _on_reference_file_dropped(self, path: str) -> None:
        """Yakalananlar galerisinden (ya da herhangi bir yerel dosyadan) canvas'a sürükleyip
        bırakılan bir görüntüyü `_on_load_references` ile AYNI şekilde galeriye ekler ve
        aktif referans yapar."""
        try:
            image = load_image(path)
        except FileNotFoundError as exc:
            QMessageBox.critical(self, "Görüntü Yüklenemedi", str(exc))
            return
        index = self._add_reference_to_gallery(path, image)
        self._activate_reference(index)

    def _add_reference_to_gallery(self, path: str, image: np.ndarray) -> int:
        index = len(self._reference_images)
        self._reference_images.append(image)
        self._reference_names.append(Path(path).name)
        item = QListWidgetItem(self._reference_names[index])
        item.setData(_REFERENCE_INDEX_ROLE, index)
        item.setIcon(QIcon(self._reference_thumbnail(image)))
        self._reference_list.addItem(item)
        return index

    def _reference_thumbnail(self, image: np.ndarray) -> QPixmap:
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        return QPixmap.fromImage(qimage).scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _on_reference_gallery_item_clicked(self, item: QListWidgetItem) -> None:
        index = item.data(_REFERENCE_INDEX_ROLE)
        if index is None or index >= len(self._reference_images):
            return
        self._activate_reference(index)

    def _activate_reference(self, index: int) -> None:
        """Galeride TEK bir görüntüyü aktif eğitim referansı yapar ve bunu listede AÇIKÇA
        ('★' öneki + `_active_reference_label`) işaretler — kullanıcı isteği: "model eğitme
        kısmında açının doğru alınması için bir tane referans seçelim". `LensCalibrationDialog`
        'daki '★ REFERANS' işaretleme deseniyle AYNI: birden fazla foto arasında hangisinin
        şu an modelin 'açı=0' baz aldığı poz olduğu belirsiz kalmasın diye."""
        self._active_reference_index = index
        self._use_reference_image(self._reference_images[index])
        self._active_reference_label.setText(f"Aktif referans: {self._reference_names[index]}")
        for i in range(self._reference_list.count()):
            item = self._reference_list.item(i)
            item_index = item.data(_REFERENCE_INDEX_ROLE)
            prefix = "★ " if item_index == index else ""
            item.setText(f"{prefix}{self._reference_names[item_index]}")

    def _use_reference_image(self, image: np.ndarray) -> None:
        self._reference_image = image
        self._canvas.set_image(image)
        x, y, w, h = self._roi
        self._canvas.set_roi(x, y, w, h)
        self._train_status_label.setText("")

    def _on_train(self) -> None:
        if self._reference_image is None:
            QMessageBox.warning(self, "Referans Görüntü Gerekli", "Önce bir referans görüntü yükleyin.")
            return

        x, y, w, h = self._roi
        roi = RoiRect(x=x, y=y, w=w, h=h)
        try:
            model = train_shape_model(
                self._reference_image,
                roi,
                num_levels=self._num_levels_spin.value(),
                min_gradient_fraction=self._min_gradient_spin.value(),
            )
        except ShapeMatchingError as exc:
            self._train_status_label.setText(f"Eğitim başarısız: {exc}")
            return

        self._model = model
        point_counts = ", ".join(str(lvl.points.shape[0]) for lvl in model.levels)
        self._train_status_label.setText(f"Model eğitildi (seviye başına nokta sayısı: {point_counts}).")
        self._save_button.setEnabled(True)
        self.model_trained.emit(model)

    def _on_save(self) -> None:
        if self._model is None:
            return
        name = self._name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Model Adı Gerekli", "Kaydetmek için bir model adı girin.")
            return

        shape_model_store.save_shape_model(name, self._model, directory=self._model_dir)
        self._save_status_label.setText(f"Kaydedildi: '{name}'")
        self._refresh_model_combo()
        self.models_changed.emit()

    def _refresh_model_combo(self) -> None:
        current = self._load_combo.currentText()
        self._load_combo.clear()
        names = shape_model_store.list_shape_models(self._model_dir)
        self._load_combo.addItems(names)
        if current in names:
            self._load_combo.setCurrentText(current)

    def _on_load_model(self) -> None:
        name = self._load_combo.currentText().strip()
        if not name:
            return
        try:
            model = shape_model_store.load_shape_model(name, directory=self._model_dir)
        except shape_model_store.ShapeModelNotFoundError as exc:
            self._save_status_label.setText(str(exc))
            return

        self._model = model
        self._save_button.setEnabled(True)
        self._name_edit.setText(name)
        self._save_status_label.setText(f"Yüklendi: '{name}'")
        self.model_trained.emit(model)

    def _on_delete_model(self) -> None:
        name = self._load_combo.currentText().strip()
        if not name:
            return
        reply = QMessageBox.question(
            self, "Modeli Sil", f"'{name}' adlı model kalıcı olarak silinsin mi?"
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        shape_model_store.delete_shape_model(name, directory=self._model_dir)
        if self._name_edit.text().strip() == name:
            self._name_edit.clear()
            self._model = None
            self._save_button.setEnabled(False)
        self._save_status_label.setText(f"Silindi: '{name}'")
        self._refresh_model_combo()
        self.models_changed.emit()

    def _on_rename_model(self) -> None:
        old_name = self._load_combo.currentText().strip()
        if not old_name:
            return
        new_name, ok = QInputDialog.getText(self, "Yeniden Adlandır", "Yeni ad:", text=old_name)
        new_name = new_name.strip()
        if not ok or not new_name or new_name == old_name:
            return

        try:
            shape_model_store.rename_shape_model(old_name, new_name, directory=self._model_dir)
        except (shape_model_store.ShapeModelNotFoundError, FileExistsError) as exc:
            QMessageBox.critical(self, "Yeniden Adlandırma Başarısız", str(exc))
            return

        if self._name_edit.text().strip() == old_name:
            self._name_edit.setText(new_name)
        self._save_status_label.setText(f"'{old_name}' -> '{new_name}' olarak yeniden adlandırıldı.")
        self._refresh_model_combo()
        self.models_changed.emit()
