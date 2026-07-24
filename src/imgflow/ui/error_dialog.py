"""Uygulama genelinde tutarlı hata gösterimi: kullanıcının kendi hatası mı (girdi/ayar
sorunu, düzeltilebilir) yoksa beklenmeyen bir uygulama hatası mı olduğunu ayırt eder.

`custom_filters.py`/`CustomFilterError` deseninin (anlaşılır, eyleme dönük Türkçe mesaj)
genele yayılmış hali. Bilinen/eyleme-dönük exception tipleri (`_ACTIONABLE_TYPES`) zaten bu
kod tabanında anlaşılır mesaj taşıyacak şekilde fırlatılıyor (`LensCalibrator`,
`HeightScaleModel`, `camera_source`, `custom_filters`, reçete/graph) — bunlar "girdi/ayar
sorunu" çerçevesiyle gösterilir. Bunların dışında kalan (ör. `TypeError`, `AttributeError` —
beklenmeyen programlama hataları) farklı, "muhtemelen bir uygulama hatası" çerçevesiyle
gösterilir ki kullanıcı hangisiyle karşı karşıya olduğunu anlayabilsin.
"""

from __future__ import annotations

from PySide6.QtWidgets import QMessageBox

from imgflow.core.errors import ImgflowError

_ACTIONABLE_TYPES = (ImgflowError, ValueError, RuntimeError, KeyError, OSError)


def show_error(parent, title: str, exc: Exception, *, hint: str = "") -> None:
    """title: dialog başlığı. exc: yakalanan exception. hint: duruma özel, ek düzeltme
    önerisi — verilirse ana mesaj olur, `exc`'in ham metni "teknik ayrıntı" olarak eklenir
    (ör. çıplak bir `KeyError`'ın anlamsız `repr` metni yerine)."""
    if isinstance(exc, _ACTIONABLE_TYPES):
        primary = hint if hint else str(exc)
        lines = [primary]
        if hint and str(exc) and str(exc) != hint:
            lines.append(f"Teknik ayrıntı: {exc}")
        lines.append("(Bu genelde bir girdi/ayar sorunudur, uygulama hatası değildir.)")
        QMessageBox.critical(parent, title, "\n".join(lines))
    else:
        QMessageBox.critical(
            parent,
            title,
            "Beklenmeyen bir hata oluştu (muhtemelen bir uygulama hatası).\n"
            f"Teknik ayrıntı: {exc}\n"
            "Ne yaptığınızı not alıp bildirirseniz sorunu çözmemize yardımcı olur.",
        )
