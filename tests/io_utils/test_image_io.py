import numpy as np

from imgflow.io_utils.image_io import load_image, save_image


def test_save_and_load_roundtrip(tmp_path):
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    img[:, :, 1] = 128
    path = tmp_path / "out.png"

    save_image(path, img)
    loaded = load_image(path)

    assert loaded.shape == img.shape
    assert (loaded == img).all()
