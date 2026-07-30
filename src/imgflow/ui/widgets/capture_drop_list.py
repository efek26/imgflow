"""Yakalananlar galerisinden (`CaptureGalleryPanel`) ya da herhangi bir yerel dosyadan
sürüklenip bırakılan kareleri kabul eden `QListWidget`.

`LensCalibrationDialog`'da tanımlanmıştı, `HeightScaleCalibrationDialog`'un da AYNI galeri
sürükle-bırak davranışına ihtiyaç duyması üzerine buraya çıkarıldı -- canlı kamera olmadan
da (ör. bant durduğunda daha önce yakalanmış zor bir kareyi) bir kalibrasyon/eğitim setine
eklemeyi sağlar. `files_dropped` bırakılan tüm yerel dosya yollarını taşır; iç öğe taşıma
davranışı DOKUNULMADAN kalır.
"""

from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QListWidget


class CaptureDropListWidget(QListWidget):
    files_dropped = Signal(list)  # list[str]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt override
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt override
        # QAbstractItemView'ın varsayılan dragMoveEvent'i kendi iç taşıma mime tipine
        # (application/x-qabstractitemmodeldatalist) göre canDrop() kontrolü yapar ve
        # text/uri-list (gerçek dosya) sürüklemelerini move sırasında reddedebilir --
        # dragEnterEvent kabul etse bile bu, bırakmayı güvenilmez/rastgele başarısız hale
        # getiriyordu (gerçek kullanıcı raporu). dragEnterEvent ile AYNI kontrolü burada da
        # yaparak move boyunca kabulü koruyoruz.
        if any(url.isLocalFile() for url in event.mimeData().urls()):
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt override
        paths = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if paths:
            self.files_dropped.emit(paths)
            event.acceptProposedAction()
        else:
            super().dropEvent(event)
