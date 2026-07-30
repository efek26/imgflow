"""Sağ panelde göze çarpan, kategori bazlı akordeon (QToolBox) şeklinde kamera ayarları aracı.

Eskiden ayrı bir non-modal QDialog'du; kullanıcı canlı önizlemeyi görmeye devam ederken
kamera ayarlarını değiştirebilsin diye ana pencerenin sağ sütununa (Parametreler'in
yanına, ikinci bir sekme olarak) taşındı.

Performans notu: her alan değişiminde TÜM kategoriyi donanımdan yeniden sorgulamak (eski
davranış) her biri ayrı bir donanım round-trip'i olan ~5-10 sorguya yol açıyordu — bir
slider sürüklenirken bu gözle görülür bir gecikmeye neden oluyordu. Bunun yerine:
  1. Hızlı yol (çoğunluk parametre): tek `controller.set()` çağrısı (1 yazma + 1 okuma),
     sadece o widget'ın gösterilen değeri güncellenir (`ParamForm.set_value`) — kategori
     yeniden kurulmaz.
  2. Yavaş yol (SADECE `is_structural()` True olan node'lar, ör. TriggerMode/UserSetSelector):
     bu tür bir node değiştiğinde başka node'ların availability/min-max'i gerçekten
     değişebileceği için tüm kategori yeniden sorgulanır/kurulur.
  3. Debounce: bir slider sürüklenirken `valueChanged` art arda onlarca kez tetiklenir;
     bunların HER birini donanıma yazmak yerine, değişiklikler `_DEBOUNCE_MS` süre
     bekletilip sadece son değer donanıma uygulanır.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import (
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from imgflow.core.camera_params import CATEGORIES, CameraParameterController
from imgflow.io_utils import camera_settings_store
from imgflow.ui.widgets.param_form import ParamForm

_DEBOUNCE_MS = 120

_CATEGORY_ICONS = {
    "Görüntü Kalitesi": "🎨",
    "Görüntü Formatı": "🖼️",
    "Yakalama Kontrolü": "⏱️",
    "Dijital I/O": "🔌",
    "Aktarım Katmanı": "🌐",
    "Kullanıcı Setleri": "💾",
}

_DISCONNECTED_TEXT = "🔴 Kamera bağlı değil"

_CATEGORY_DESCRIPTIONS = {
    "Görüntü Kalitesi": "Pozlama, kazanç, siyah seviyesi ve beyaz dengesi — görüntünün "
    "parlaklığını/renk doğruluğunu belirler.",
    "Görüntü Formatı": "Piksel formatı, çözünürlük (Width/Height/Offset), binning/decimation "
    "ve aynalama — kameranın ne boyutta/hangi kodlamada görüntü göndereceğini belirler.",
    "Yakalama Kontrolü": "Kare hızı ve tetikleme (Trigger) ayarları — kameranın kendi "
    "hızında mı yoksa harici bir sinyalle (ör. PLC/konveyör) mi kare üreteceğini belirler.",
    "Dijital I/O": "Kameranın üzerindeki fiziksel giriş/çıkış pinlerini (Line1, Line2, ...) "
    "yapılandırır — ör. harici tetik girişi ya da flaş/strobe çıkışı olarak kullanmak için.",
    "Aktarım Katmanı": "Ağ üzerinden veri aktarım ayarları (paket boyutu, gecikme, bant "
    "genişliği). Çoğu alan sadece GigE kameralarda görünür; USB3 kameralarda bu alanlar "
    "genelde mevcut değildir (gizli kalır).",
    "Kullanıcı Setleri": "Kamera belleğinde (donanımda) saklı ayar profilleri — mevcut "
    "ayarları bir profile kaydetme ve kamera açılışında hangi profilin otomatik "
    "yükleneceğini seçme.",
}


class CameraSettingsPanel(QWidget):
    def __init__(
        self,
        parent: QWidget | None = None,
        camera_settings_dir: str | Path | None = None,
    ) -> None:
        super().__init__(parent)
        self._controller: CameraParameterController | None = None
        self._forms: dict[str, ParamForm] = {}
        self._pending_changes: dict[str, tuple[str, Any]] = {}
        self._camera_settings_dir = (
            camera_settings_dir if camera_settings_dir is not None else camera_settings_store.DEFAULT_CAMERA_SETTINGS_DIR
        )

        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.timeout.connect(self._flush_pending_changes)

        self._status_label = QLabel(_DISCONNECTED_TEXT)
        self._status_label.setStyleSheet("font-weight: 600; padding: 2px;")
        # `setWordWrap` YOKTU -- kamera bağlanınca (ör. önceki ayarlar otomatik geri
        # yüklendiğinde `main_window.py::start_camera`'nın eklediği " — önceki ayarlar geri
        # yüklendi" son ekiyle) metin uzayınca, sarmasız bir QLabel'ın minimumSizeHint'i
        # TÜM metnin genişliğine eşit olur -- bu da yukarı doğru (panel -> sekme -> splitter
        # -> ana pencere) taşınıp kullanıcı hiçbir şey yapmadan ana pencereyi zorla
        # büyütüyordu (gerçek kullanıcı raporu: "gereksiz boyut değiştirme"). `setWordWrap`
        # TEK BAŞINA yeterli DEĞİL -- metindeki tek bir uzun/boşluksuz "kelime" (ör. parantez
        # içi sınıf adı "(BaslerCameraSource)") sarmayla bile bölünemediğinden minimumSizeHint
        # hâlâ belirgin genişlikte kalıyordu (ölçülerek doğrulandı). Asıl çözüm yatay
        # `QSizePolicy.Ignored`: layout bu widget'ın sizeHint/minimumSizeHint'ini TAMAMEN yok
        # sayar, etiket mevcut alana göre sarar/kırpılır ama ASLA panelin (dolayısıyla ana
        # pencerenin) minimum genişliğini büyütmez.
        self._status_label.setWordWrap(True)
        self._status_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        refresh_button = QPushButton("⟳ Yenile")
        refresh_button.setToolTip("Tüm değerleri kameradan yeniden oku")
        refresh_button.clicked.connect(self.refresh)

        self._save_settings_button = QPushButton("💾 Kaydet")
        self._save_settings_button.setToolTip(
            "Kameranın şu anki tüm ayarlarını isimli bir dosya ön ayarı olarak kaydeder "
            "(kameranın kendi UserSet belleğinden FARKLI — bu diskte bir dosyadır, başka "
            "bir oturuma/kameraya taşınabilir)."
        )
        self._save_settings_button.setEnabled(False)
        self._save_settings_button.clicked.connect(self._on_save_settings)

        self._load_settings_button = QPushButton("📂 Yükle")
        self._load_settings_button.setToolTip("Daha önce kaydedilmiş bir ayar ön ayarını kameraya uygular.")
        self._load_settings_button.setEnabled(False)
        self._load_settings_button.clicked.connect(self._on_load_settings)

        header = QHBoxLayout()
        header.addWidget(self._status_label, 1)
        header.addWidget(refresh_button)
        header.addWidget(self._save_settings_button)
        header.addWidget(self._load_settings_button)

        self._io_status_label = QLabel("")
        self._io_status_label.setWordWrap(True)
        self._io_status_label.setStyleSheet("color: #2e7d32;")

        self._error_label = QLabel("")
        """Donanım yazma/komut hatalarını gösterir. KASITLI OLARAK modal bir QMessageBox
        DEĞİL, satır içi bir etiket: bu panelin donanıma yazma yolu debounce zamanlayıcısı
        üzerinden çalışıyor — bir alan sürekli reddedilirse (ör. TriggerMode kamerayı geçici
        bir duruma soktuysa) modal bir diyalog her tetiklemede yeniden açılıp kullanıcının
        uygulamayı kapatmasını bile engelleyebilir (gerçek bir kullanıcı raporuyla
        doğrulandı) — satır içi etiket asla bloke etmez/yığılmaz."""
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c0392b;")

        self._toolbox = QToolBox()

        # Gerçek kullanıcı raporu: kamera bağlanıp `set_controller` gerçek GenICam
        # kategorileriyle `_toolbox`'ı doldurunca (bkz. aşağıdaki `set_controller`), panel
        # HİÇBİR sekme değişimi olmadan bile ana penceredeki `central_splitter`'ı zorlayıp
        # TÜM pencereyi (kullanıcı hiç isteMEDEN) büyütüyordu -- `_on_right_tabs_changed`
        # (bkz. `main_window.py`) SADECE sekme DEĞİŞİNCE devreye girdiğinden, kullanıcı zaten
        # "Kamera Ayarları" sekmesindeyken kamerayı bağlarsa bu korumadan hiç faydalanmıyordu.
        # Kök neden: `QToolBox` doğrudan bu düzene eklenmişti -- sayfalar dolunca büyüyen
        # minimum boyut isteği ÖZÜMSENMEDEN doğrudan yukarı (panel -> sekme -> splitter ->
        # ana pencere) taşınıyordu. `ShapeMatchingDialog`'un kontrol panelini `QScrollArea`'ya
        # sarmalayan AYNI çözüm: `_toolbox` artık `setWidgetResizable(True)` ile bir
        # `QScrollArea` içinde -- içerik ne kadar büyürse büyüsün dışarıya sadece KENDİ
        # kaydırma çubuğuyla taşar, ana pencereyi asla zorlamaz.
        toolbox_scroll = QScrollArea()
        toolbox_scroll.setWidgetResizable(True)
        toolbox_scroll.setWidget(self._toolbox)

        layout = QVBoxLayout(self)
        layout.addLayout(header)
        layout.addWidget(self._error_label)
        layout.addWidget(self._io_status_label)
        layout.addWidget(toolbox_scroll, 1)

    # -- genel API ------------------------------------------------------

    @property
    def settings_dir(self) -> str | Path:
        return self._camera_settings_dir

    def current_values(self) -> dict[str, Any]:
        return self._current_all_values()

    def set_controller(self, controller: CameraParameterController | None, status_text: str | None = None) -> None:
        self._pending_changes.clear()
        self._debounce_timer.stop()
        self._error_label.setText("")
        self._controller = controller
        self._forms.clear()
        while self._toolbox.count():
            widget = self._toolbox.widget(0)
            self._toolbox.removeItem(0)
            widget.deleteLater()

        self._save_settings_button.setEnabled(controller is not None)
        self._load_settings_button.setEnabled(controller is not None)

        if controller is None:
            self._status_label.setText(_DISCONNECTED_TEXT)
            return

        self._status_label.setText(status_text or "🟢 Kamera bağlı")
        for category in CATEGORIES:
            specs = controller.build_specs(category)
            commands = controller.command_defs(category)
            if not specs and not commands:
                continue
            page = self._build_category_page(category, specs, commands)
            icon = _CATEGORY_ICONS.get(category, "")
            self._toolbox.addItem(page, f"{icon} {category}".strip())

    def refresh(self) -> None:
        if self._controller is not None:
            self.set_controller(self._controller, self._status_label.text())

    # -- inşa -------------------------------------------------------------

    def _build_category_page(self, category: str, specs, commands) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        description = _CATEGORY_DESCRIPTIONS.get(category, "")
        if description:
            description_label = QLabel(description)
            description_label.setWordWrap(True)
            description_label.setStyleSheet("color: #666; font-style: italic;")
            layout.addWidget(description_label)

        form = ParamForm()
        form.set_params(specs, self._controller.current_values(specs))
        form.field_changed.connect(lambda name, value, cat=category: self._on_field_changed(cat, name, value))
        self._forms[category] = form
        layout.addWidget(form)

        for command_def in commands:
            button = QPushButton(command_def.label)
            button.clicked.connect(lambda _checked=False, name=command_def.node_name: self._on_execute(name))
            layout.addWidget(button)

        layout.addStretch(1)
        return page

    # -- olaylar ----------------------------------------------------------

    def _on_field_changed(self, category: str, name: str, value: Any) -> None:
        self._pending_changes[name] = (category, value)
        self._debounce_timer.start(_DEBOUNCE_MS)

    def _flush_pending_changes(self) -> None:
        if self._controller is None:
            self._pending_changes.clear()
            return
        pending, self._pending_changes = self._pending_changes, {}
        for name, (category, value) in pending.items():
            try:
                actual = self._controller.set(name, value)
            except Exception as exc:
                # Kamera bağlantısı kesilmiş ya da kamera değeri reddetmiş olabilir — pypylon'un
                # GenICam exception hiyerarşisi bu ortamda doğrulanamadığı için (pypylon kurulu
                # değil) tip bazlı ayrım YAPILMIYOR, geniş bir Exception yakalanıyor (hiç
                # yakalamamaktan kesinlikle daha iyi — önceden bu QTimer callback'inde
                # yakalanmadan patlıyordu).
                self._error_label.setText(
                    f"'{name}' ayarı uygulanamadı: {exc} — kamera bağlantısının kesilmemiş "
                    "olduğundan ve değerin izin verilen aralıkta olduğundan emin olun."
                )
                self._safe_rebuild_category(category)  # panel donanımla senkron kalsın diye tazele
                continue
            self._error_label.setText("")
            if self._controller.is_structural(name):
                self._safe_rebuild_category(category)
            else:
                form = self._forms.get(category)
                if form is not None:
                    form.set_value(name, actual)

    def _rebuild_category(self, category: str) -> None:
        form = self._forms.get(category)
        if form is None or self._controller is None:
            return
        specs = self._controller.build_specs(category)
        form.set_params(specs, self._controller.current_values(specs))

    def _safe_rebuild_category(self, category: str) -> None:
        """`_rebuild_category` de (ör. az önce yazılamayan alanın etkilediği bir komşu
        node'u okurken) hata verirse bu istisna YAKALANMAZSA `_flush_pending_changes`'i
        çağıran QTimer slotundan kaçıp uygulama genelindeki `sys.excepthook`'a düşer — bu da
        (tekrarlayan bir zamanlayıcı bağlamında) yeni bir modal diyalog fırtınası riski
        taşır. Bu yüzden burada da yakalanıp satır içi etikette gösterilir, ASLA modal
        gösterilmez."""
        try:
            self._rebuild_category(category)
        except Exception as exc:
            self._error_label.setText(f"Kategori tazelenirken hata: {exc}")

    def _on_execute(self, name: str) -> None:
        if self._controller is None:
            return
        try:
            self._controller.execute(name)
        except Exception as exc:
            self._error_label.setText(f"'{name}' komutu çalıştırılamadı: {exc}")
        else:
            self._error_label.setText("")

    # -- dosya bazlı ayar ön ayarları (kaydet/yükle) -----------------------

    def _current_all_values(self) -> dict[str, Any]:
        if self._controller is None:
            return {}
        values: dict[str, Any] = {}
        for category in CATEGORIES:
            specs = self._controller.build_specs(category)
            values.update(self._controller.current_values(specs))
        return values

    def _on_save_settings(self) -> None:
        if self._controller is None:
            return
        existing = camera_settings_store.list_settings(self._camera_settings_dir)
        name, ok = QInputDialog.getItem(self, "Ayarları Kaydet", "Profil adı:", existing, 0, True)
        name = name.strip()
        if not ok or not name:
            return
        camera_settings_store.save_settings(name, self._current_all_values(), directory=self._camera_settings_dir)
        self._io_status_label.setText(f"Kaydedildi: '{name}'")

    def _on_load_settings(self) -> None:
        if self._controller is None:
            return
        names = camera_settings_store.list_settings(self._camera_settings_dir)
        if not names:
            self._io_status_label.setText("Kayıtlı kamera ayarı ön ayarı yok.")
            return
        name, ok = QInputDialog.getItem(self, "Ayarları Yükle", "Profil adı:", names, 0, False)
        if not ok or not name:
            return
        try:
            data = camera_settings_store.load_settings(name, directory=self._camera_settings_dir)
        except (OSError, ValueError, KeyError) as exc:
            self._io_status_label.setText(f"'{name}' yüklenemedi: {exc}")
            return

        applied = 0
        skipped = 0
        for param_name, value in data.values.items():
            try:
                self._controller.set(param_name, value)
            except Exception:
                skipped += 1
            else:
                applied += 1

        self.set_controller(self._controller, self._status_label.text())
        self._io_status_label.setText(
            f"Yüklendi: '{name}' ({applied} ayar uygulandı, {skipped} atlandı)"
        )
