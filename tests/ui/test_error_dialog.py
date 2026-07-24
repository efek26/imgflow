from imgflow.core.errors import ImgflowError
from imgflow.ui.error_dialog import show_error


def _shown(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "imgflow.ui.error_dialog.QMessageBox.critical",
        lambda parent, title, text: calls.append((title, text)),
    )
    return calls


def test_known_actionable_exception_shows_message_directly(monkeypatch):
    calls = _shown(monkeypatch)

    show_error(None, "Başlık", ValueError("En az 2 kare gerekli."))

    assert len(calls) == 1
    title, text = calls[0]
    assert title == "Başlık"
    assert "En az 2 kare gerekli." in text
    assert "girdi/ayar sorunu" in text


def test_hint_becomes_primary_message_and_raw_exception_is_secondary(monkeypatch):
    calls = _shown(monkeypatch)

    show_error(None, "Başlık", KeyError("id"), hint="Reçete dosyası bozuk görünüyor.")

    _title, text = calls[0]
    assert text.startswith("Reçete dosyası bozuk görünüyor.")
    assert "Teknik ayrıntı" in text


def test_imgflow_error_is_actionable(monkeypatch):
    calls = _shown(monkeypatch)

    show_error(None, "Başlık", ImgflowError("özel hata"))

    _title, text = calls[0]
    assert "özel hata" in text
    assert "girdi/ayar sorunu" in text


def test_unexpected_exception_gets_app_bug_framing(monkeypatch):
    calls = _shown(monkeypatch)

    show_error(None, "Başlık", TypeError("beklenmedik"))

    _title, text = calls[0]
    assert "muhtemelen bir uygulama hatası" in text
    assert "beklenmedik" in text


def test_hint_not_duplicated_when_it_matches_exception_text(monkeypatch):
    calls = _shown(monkeypatch)

    show_error(None, "Başlık", ValueError("aynı metin"), hint="aynı metin")

    _title, text = calls[0]
    assert text.count("aynı metin") == 1
