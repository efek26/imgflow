import sys

import pytest

from imgflow.cli import _install_excepthook


@pytest.fixture(autouse=True)
def _restore_excepthook():
    original = sys.excepthook
    yield
    sys.excepthook = original


def _raise_and_hook(exc):
    try:
        raise exc
    except type(exc):
        sys.excepthook(*sys.exc_info())


def test_excepthook_shows_error_for_first_occurrence(monkeypatch):
    calls = []
    monkeypatch.setattr("imgflow.ui.error_dialog.show_error", lambda *a, **k: calls.append(a))
    _install_excepthook()

    _raise_and_hook(RuntimeError("boom"))

    assert len(calls) == 1


def test_excepthook_suppresses_repeat_of_same_error_within_window(monkeypatch):
    """Kritik regresyon testi: bir zamanlayıcı (ör. kamera tick'i) aynı istisnayı arka
    arkaya fırlatırsa, bu son-çare ağının kendisi bir modal diyalog fırtınası kaynağına
    dönüşmemeli."""
    calls = []
    monkeypatch.setattr("imgflow.ui.error_dialog.show_error", lambda *a, **k: calls.append(a))
    _install_excepthook()

    for _ in range(5):
        _raise_and_hook(RuntimeError("tekrarlayan hata"))

    assert len(calls) == 1


def test_excepthook_shows_different_errors_separately(monkeypatch):
    calls = []
    monkeypatch.setattr("imgflow.ui.error_dialog.show_error", lambda *a, **k: calls.append(a))
    _install_excepthook()

    _raise_and_hook(RuntimeError("hata A"))
    _raise_and_hook(ValueError("hata B"))

    assert len(calls) == 2
