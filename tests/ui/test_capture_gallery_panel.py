from pathlib import Path

import numpy as np
import pytest
from PySide6.QtWidgets import QAbstractItemView

from imgflow.core import capture_store
from imgflow.ui.panels.capture_gallery_panel import CaptureGalleryPanel, _PATH_ROLE


@pytest.fixture(autouse=True)
def _isolated_capture_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(capture_store, "CAPTURE_DIR", tmp_path / "captures")


def _image() -> np.ndarray:
    return np.zeros((4, 4, 3), dtype=np.uint8)


def test_panel_starts_empty_without_explicit_refresh(qtbot):
    capture_store.save_capture(_image(), source="lens")

    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)

    assert panel._list.count() == 0


def test_refresh_populates_list_from_store(qtbot):
    capture_store.save_capture(_image(), source="lens")
    capture_store.save_capture(_image(), source="height_scale")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)

    panel.refresh()

    assert panel._list.count() == 2


def test_delete_selected_removes_from_store_and_list(qtbot):
    capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    panel._list.setCurrentRow(0)

    panel._on_delete_selected()

    assert panel._list.count() == 0
    assert capture_store.list_captures() == []


def test_delete_selected_with_no_selection_is_a_no_op(qtbot):
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)

    panel._on_delete_selected()  # patlamamalı

    assert panel._list.count() == 0


def test_list_supports_multi_selection(qtbot):
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)

    assert panel._list.selectionMode() == QAbstractItemView.SelectionMode.ExtendedSelection


def test_delete_selected_removes_multiple_items_at_once(qtbot):
    for _ in range(3):
        capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    for row in range(3):
        panel._list.item(row).setSelected(True)

    panel._on_delete_selected()

    assert panel._list.count() == 0
    assert capture_store.list_captures() == []


def test_delete_selected_only_removes_the_selected_ones(qtbot):
    for _ in range(3):
        capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    panel._list.item(0).setSelected(True)
    panel._list.item(1).setSelected(True)

    panel._on_delete_selected()

    assert panel._list.count() == 1
    assert len(capture_store.list_captures()) == 1


def test_export_selected_copies_files_to_chosen_directory(qtbot, tmp_path, monkeypatch):
    for _ in range(2):
        capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    for row in range(2):
        panel._list.item(row).setSelected(True)

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    monkeypatch.setattr(
        "imgflow.ui.panels.capture_gallery_panel.QFileDialog.getExistingDirectory",
        lambda *a, **k: str(export_dir),
    )
    monkeypatch.setattr("imgflow.ui.panels.capture_gallery_panel.QMessageBox.information", lambda *a, **k: None)

    panel._on_export_selected()

    exported_files = list(export_dir.iterdir())
    assert len(exported_files) == 2
    # orijinal kayıtlar silinmeden korunmalı (dışa aktarma KOPYALAR, taşımaz)
    assert len(capture_store.list_captures()) == 2


def test_export_selected_with_no_selection_is_a_no_op(qtbot, monkeypatch):
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)

    called = []
    monkeypatch.setattr(
        "imgflow.ui.panels.capture_gallery_panel.QFileDialog.getExistingDirectory",
        lambda *a, **k: called.append(1),
    )

    panel._on_export_selected()  # patlamamalı

    assert called == []


def test_list_items_carry_capture_path_for_drag_and_drop(qtbot):
    """Galerideki her item, gerçek dosya yolunu (`_PATH_ROLE`) taşımalı — sürükle-bırak,
    ana pipeline önizlemesine (ya da ileride başka bir hedefe) bu yolu bir dosya URL'i olarak
    aktarır (bkz. `_DraggableCaptureList.mimeData`, `ImageView.dropEvent`)."""
    capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()

    record = capture_store.list_captures()[0]
    assert panel._list.item(0).data(_PATH_ROLE) == str(record.path)


def test_list_supports_drag_out_as_file_urls(qtbot):
    capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()

    assert panel._list.dragEnabled() is True
    mime = panel._list.mimeData([panel._list.item(0)])
    assert mime.hasUrls()
    record = capture_store.list_captures()[0]
    assert Path(mime.urls()[0].toLocalFile()) == record.path


def test_export_selected_cancelled_dialog_does_nothing(qtbot, tmp_path, monkeypatch):
    capture_store.save_capture(_image(), source="lens")
    panel = CaptureGalleryPanel()
    qtbot.addWidget(panel)
    panel.refresh()
    panel._list.item(0).setSelected(True)

    monkeypatch.setattr(
        "imgflow.ui.panels.capture_gallery_panel.QFileDialog.getExistingDirectory", lambda *a, **k: ""
    )

    panel._on_export_selected()  # kullanıcı klasör seçmeden iptal etti

    # capture_store dizini dışında (fixture'ın kendi oluşturduğu) hiçbir yeni klasör/dosya oluşmamalı
    assert list(tmp_path.iterdir()) == [tmp_path / "captures"]
