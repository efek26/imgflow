"""Kalibrasyon sırasında yakalanan (ve `core/capture_store.py` ile diske kaydedilen)
karelerin küçük resim galerisi — `main_window.py`'de ayrılabilir/gizlenebilir bir
`QDockWidget` içinde gösterilir (hız önceliğinde tamamen kapatılabilsin diye)."""

from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QMimeData, Qt, QUrl, Signal
from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.capture_store import CaptureRecord, delete_capture, list_captures
from imgflow.io_utils.image_io import load_image
from imgflow.ui.widgets.image_view import normalize_to_uint8, numpy_to_qimage

_THUMB_SIZE = 96
_ID_ROLE = Qt.ItemDataRole.UserRole
_PATH_ROLE = Qt.ItemDataRole.UserRole + 1
_SOURCE_LABELS = {
    "lens": "Lens",
    "height_scale": "Yükseklik-Ölçek",
    "live": "Fotoğraf",
    "filtered": "Filtrelenmiş",
    "measurement": "Ölçüm",
    "shape_match": "Şekil Eşleştirme",
    "flat_field": "Aydınlatma Referansı",
}


class _DraggableCaptureList(QListWidget):
    """Galeriden diğer panellere (ör. ana pipeline önizlemesi, ileride kalibrasyon/şekil
    eşleştirme kare listeleri) sürükle-bırak — `QListWidget`'ın varsayılan iç-taşıma mime
    tipi yerine GERÇEK dosya URL'leri (`text/uri-list`) üretir, böylece bırakma hedefi
    sıradan bir dosya sürüklemesiymiş gibi ele alabilir (bkz. `ImageView.dropEvent`)."""

    def mimeData(self, items: list[QListWidgetItem]) -> QMimeData:  # noqa: N802 - Qt override
        urls = [QUrl.fromLocalFile(path) for item in items if (path := item.data(_PATH_ROLE))]
        mime = QMimeData()
        if urls:
            mime.setUrls(urls)
        return mime


class CaptureGalleryPanel(QWidget):
    """`refresh()` yapıcıdan (`__init__`) KASITLI OLARAK çağrılmaz — gerçek diski
    (`~/.imgflow/captures`) sadece açık bir `refresh()` çağrısıyla okur. Bu panel
    `main_window.py` üzerinden testlerde defalarca örneklenir; yapıcıda otomatik disk okuma
    testleri kullanıcının gerçek makine durumuna bağımlı kılar ve yavaşlatır (bkz.
    `custom_filters.load_all_custom_filters`'ın aynı sebeple `cli.py`'ye taşınması). Gerçek
    uygulama açılışında `cli.py` bu metodu bir kez çağırır."""

    open_requested = Signal(str)
    """Bir kare 'Pipeline'a Yükle' ile (çift tıklama ya da context menü) seçilince yayınlanır
    -- taşınan değer dosya yoludur (`_PATH_ROLE`). Gerçek kullanıcı isteği: "çektiğim
    fotoğraflarda daha sonradan uzunluk bulabilmeliyim" -- eskiden bunun TEK yolu (galeriden
    ana önizlemeye) sürükle-bırak yapmaktı, bu ikinci/keşfedilir bir giriş noktası ekler.
    `main_window.py` bunu `open_image(path)`'e bağlar (sürükle-bırak yoluyla AYNI çağrı)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._list = _DraggableCaptureList()
        self._list.setViewMode(QListWidget.ViewMode.IconMode)
        self._list.setIconSize(QPixmap(_THUMB_SIZE, _THUMB_SIZE).size())
        self._list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self._list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._list.setDragEnabled(True)
        self._list.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self._list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._on_context_menu)
        self._list.itemDoubleClicked.connect(self._on_item_double_clicked)

        delete_button = QPushButton("Seçilenleri Sil")
        delete_button.clicked.connect(self._on_delete_selected)
        export_button = QPushButton("Seçilenleri Dışa Aktar...")
        export_button.clicked.connect(self._on_export_selected)

        button_row = QHBoxLayout()
        button_row.addWidget(delete_button)
        button_row.addWidget(export_button)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._list, 1)
        layout.addLayout(button_row)

    def refresh(self) -> None:
        self._list.clear()
        for record in list_captures():
            label = f"{_SOURCE_LABELS.get(record.source, record.source)}\n{record.timestamp}"
            item = QListWidgetItem(label)
            item.setData(_ID_ROLE, record.id)
            item.setData(_PATH_ROLE, str(record.path))
            item.setIcon(QIcon(self._thumbnail(record)))
            self._list.addItem(item)

    def _thumbnail(self, record: CaptureRecord) -> QPixmap:
        try:
            image = load_image(record.path)
        except FileNotFoundError:
            return QPixmap()
        qimage = numpy_to_qimage(normalize_to_uint8(image))
        return QPixmap.fromImage(qimage).scaled(
            _THUMB_SIZE,
            _THUMB_SIZE,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )

    def _on_delete_selected(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        for item in items:
            delete_capture(item.data(_ID_ROLE))
        self.refresh()

    def _on_export_selected(self) -> None:
        items = self._list.selectedItems()
        if not items:
            return
        directory = QFileDialog.getExistingDirectory(self, "Dışa Aktarma Klasörü Seç")
        if not directory:
            return

        records_by_id = {r.id: r for r in list_captures()}
        exported = 0
        for item in items:
            record = records_by_id.get(item.data(_ID_ROLE))
            if record is None or not record.path.exists():
                continue
            shutil.copy2(record.path, Path(directory) / record.path.name)
            exported += 1
        QMessageBox.information(self, "Dışa Aktarıldı", f"{exported} kare '{directory}' klasörüne kopyalandı.")

    def _on_item_double_clicked(self, item: QListWidgetItem) -> None:
        path = item.data(_PATH_ROLE)
        if path:
            self.open_requested.emit(path)

    def _on_context_menu(self, pos) -> None:
        item = self._list.itemAt(pos)
        if item is None:
            return
        if item not in self._list.selectedItems():
            self._list.setCurrentItem(item)
        menu = QMenu(self)
        open_action = menu.addAction("Pipeline'a Yükle")
        menu.addSeparator()
        delete_action = menu.addAction("Seçilenleri Sil")
        export_action = menu.addAction("Seçilenleri Dışa Aktar...")
        action = menu.exec(self._list.mapToGlobal(pos))
        if action == open_action:
            self._on_item_double_clicked(item)
        elif action == delete_action:
            self._on_delete_selected()
        elif action == export_action:
            self._on_export_selected()
