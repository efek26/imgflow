import cv2
import numpy as np
import pytest

from imgflow.io_utils.image_io import load_image, save_image


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[:, :, 1] = 128
    path = tmp_path / "out.png"

    save_image(path, img)
    loaded = load_image(path)

    assert loaded.shape == img.shape
    assert (loaded == img).all()


def test_save_and_load_roundtrip_with_unicode_path(tmp_path):
    # Windows'ta cv2.imread/imwrite Türkçe karakter içeren yolları (ör. "Masaüstü") açamaz;
    # bu test image_io'nun bu sınırlamayı aşan Path.read_bytes/write_bytes yolunu kullandığını doğrular.
    img = np.full((4, 4, 3), 200, dtype=np.uint8)
    unicode_dir = tmp_path / "Masaüstü_görüntüler"
    path = unicode_dir / "balon_çiçek.jpg"

    save_image(path, img)
    loaded = load_image(path)

    assert loaded.shape == img.shape


def test_load_missing_file_raises_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_image(tmp_path / "yok.png")


def test_load_drops_alpha_channel(tmp_path):
    # Şeffaf PNG gibi 4 kanallı (BGRA) görüntüler pipeline'ın geri kalanını (color.convert,
    # threshold, ...) kıracağı için burada BGR'a düşürülmeli.
    rgba = np.zeros((6, 6, 4), dtype=np.uint8)
    rgba[:, :, 3] = 128  # yarı saydam
    path = tmp_path / "seffaf.png"
    ok, buffer = cv2.imencode(".png", rgba)
    assert ok
    path.write_bytes(buffer.tobytes())

    loaded = load_image(path)

    assert loaded.shape == (6, 6, 3)
