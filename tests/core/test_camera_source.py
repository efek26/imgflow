import cv2
import numpy as np
import pytest

from imgflow.core.camera_source import (
    PYLON_AVAILABLE,
    BaslerCameraSource,
    GigeCameraSource,
    UsbCameraSource,
    VideoFileSource,
)


def test_gige_camera_source_is_alias_for_basler_camera_source():
    assert GigeCameraSource is BaslerCameraSource


def test_usb_camera_invalid_index_raises():
    with pytest.raises(RuntimeError):
        UsbCameraSource(device_index=99)


def test_video_file_missing_path_raises(tmp_path):
    with pytest.raises(RuntimeError):
        VideoFileSource(str(tmp_path / "yok.avi"))


def test_video_file_reads_frames_and_loops(tmp_path):
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"XVID"), 5, (8, 8))
    frame_a = np.zeros((8, 8, 3), dtype=np.uint8)
    frame_b = np.full((8, 8, 3), 255, dtype=np.uint8)
    writer.write(frame_a)
    writer.write(frame_b)
    writer.release()

    source = VideoFileSource(str(path))
    try:
        first = source.read()
        second = source.read()
        assert first is not None
        assert second is not None
        # video 2 karelik; üçüncü okuma başa sarıp tekrar ilk kareyi vermeli (loop)
        looped = source.read()
        assert looped is not None
    finally:
        source.release()


@pytest.mark.skipif(PYLON_AVAILABLE, reason="pypylon kurulu; bu test onun kurulu OLMADIĞI durumu doğrular")
def test_gige_camera_raises_clear_error_without_pypylon():
    with pytest.raises(RuntimeError, match="pypylon"):
        GigeCameraSource()
