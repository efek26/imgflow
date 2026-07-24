"""imgflow komut satırı giriş noktası: PySide6 arayüzünü başlatır."""

from __future__ import annotations

import sys
import time
import traceback

_REPEAT_SUPPRESS_WINDOW_S = 5.0
"""Bir kamera/timer tick'i gibi tekrarlayan bir yol AYNI istisnayı arka arkaya fırlatırsa,
bu pencere içinde ikinci bir modal AÇILMAZ (sadece konsola loglanır) — sadece konsola
traceback basmaktan modal göstermeye geçilince yaşanan gerçek bir olay: TriggerMode
değişikliğinin kamerayı geçici olarak kare üretemez duruma sokması, her 100ms'lik tick'te
`_on_camera_tick`'ten kaçan aynı istisnanın yeni bir modal açmasına, kullanıcının
uygulamayı kapatamayacağı bir "diyalog fırtınası"na yol açtı. `_on_camera_tick` ve
`camera_settings_panel.py`'deki debounce yolu artık kendi hatalarını satır içi (modal
olmayan) gösteriyor, ama bu son-çare ağının kendisi de asla bir spam kaynağına
dönüşmemeli — bu yüzden burada da bir tekrar bastırma var (savunma katmanı)."""


def _install_excepthook() -> None:
    """Bir Qt slot'undan kaçan ve spesifik bir try/except tarafından yakalanmamış HERHANGİ
    bir exception için son çare ağı — önceden bu durumda kullanıcı hiçbir şey görmeden
    konsola traceback basılıp uygulama tutarsız durumda kalabiliyordu. Mevcut spesifik
    handler'ların (kamera açma, reçete, vb.) yerini almaz, sadece onları atlayanları yakalar.
    """
    _last_shown: dict[tuple[type, str], float] = {}

    def _handle(exc_type, exc_value, exc_tb) -> None:
        traceback.print_exception(exc_type, exc_value, exc_tb)
        key = (exc_type, str(exc_value))
        now = time.monotonic()
        if now - _last_shown.get(key, -_REPEAT_SUPPRESS_WINDOW_S) < _REPEAT_SUPPRESS_WINDOW_S:
            return  # aynı hata kısa süre önce zaten gösterildi — tekrar modal açma
        _last_shown[key] = now

        from imgflow.ui.error_dialog import show_error

        show_error(None, "Beklenmeyen Hata", exc_value)

    sys.excepthook = _handle


def main() -> int:
    from PySide6.QtWidgets import QApplication, QMessageBox

    from imgflow.core.custom_filters import load_all_custom_filters
    from imgflow.ui.main_window import MainWindow

    _install_excepthook()
    app = QApplication(sys.argv)
    # Disk taramasını burada (uygulama başlangıcında) yapıyoruz, MainWindow.__init__ içinde
    # DEĞİL — MainWindow test paketinde defalarca örneklenir, her seferinde gerçek
    # ~/.imgflow/custom_filters dizinini taramak testleri kullanıcının makine durumuna
    # bağımlı kılardı.
    failures = load_all_custom_filters()
    window = MainWindow()
    window.resize(1000, 700)
    window.capture_gallery_panel.refresh()  # aynı sebeple burada: bkz. CaptureGalleryPanel docstring'i
    window.show()
    if failures:
        names = ", ".join(defn.name for defn, _exc in failures)
        QMessageBox.warning(
            window,
            "Bazı Özel Filtreler Yüklenemedi",
            f"Şu özel filtreler bozuk olduğu için yüklenemedi: {names}\n"
            "'+ Özel Filtre Ekle/Düzenle...' penceresinden düzeltebilir ya da silebilirsiniz.",
        )
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
