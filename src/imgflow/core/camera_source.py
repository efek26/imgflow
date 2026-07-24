"""USB kamera, video dosyası ve GigE/Basler kamera kaynaklarını ortak bir arayüz altında toplar.

pypylon (Basler GigE/USB3 SDK'sı) kurulu değilse GigeCameraSource kullanılmaya çalışıldığında
sadece anlaşılır bir RuntimeError fırlatılır — import hatası uygulamanın geri kalanını
etkilemez, sadece o kaynak türü kullanılamaz.
"""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np

from imgflow.core.genicam_node_map import GenicamNodeMap, PylonNodeMap

try:
    from pypylon import pylon

    PYLON_AVAILABLE = True
except ImportError:  # pragma: no cover - pypylon donanım SDK'sı, test ortamında kurulu değil
    pylon = None
    PYLON_AVAILABLE = False


class CameraSource(Protocol):
    def read(self) -> np.ndarray | None: ...

    def release(self) -> None: ...


class UsbCameraSource:
    """Yerleşik/USB webcam ya da endüstriyel USB kamera (OpenCV VideoCapture üzerinden)."""

    def __init__(self, device_index: int = 0) -> None:
        self._cap = cv2.VideoCapture(device_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"Kamera açılamadı (index={device_index}).")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class VideoFileSource:
    """Bir video dosyasını kare kare oynatır; sona gelince başa sarar (loop)."""

    def __init__(self, path: str) -> None:
        self._cap = cv2.VideoCapture(str(path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Video dosyası açılamadı: {path}")

    def read(self) -> np.ndarray | None:
        ok, frame = self._cap.read()
        if not ok:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        self._cap.release()


class BaslerCameraSource:
    """Basler GigE/USB3 kamera (pypylon). GrabStrategy_LatestImageOnly kullanır ki işleme
    yetişemediğinde buffer birikip gecikme büyümesin — her zaman en güncel kare alınır.

    GigE ve USB3 Vision Basler kameraları pypylon'da aynı `InstantCamera`/GenICam API'si
    üzerinden erişilir (transport farkı `TlFactory.EnumerateDevices()` seviyesinde
    soyutlanır) — bu sınıf ikisini de kapsar, isim artık buna göre.
    """

    def __init__(self, camera_index: int = 0) -> None:
        if not PYLON_AVAILABLE:
            raise RuntimeError(
                "pypylon kurulu değil; GigE/Basler kamera desteği için 'pip install pypylon' gerekir."
            )
        factory = pylon.TlFactory.GetInstance()
        devices = factory.EnumerateDevices()
        if not devices:
            raise RuntimeError("Hiçbir GigE/Basler kamera bulunamadı.")
        if camera_index >= len(devices):
            raise RuntimeError(f"Kamera index'i {camera_index} bulunamadı ({len(devices)} kamera mevcut).")

        self._camera = pylon.InstantCamera(factory.CreateDevice(devices[camera_index]))
        self._camera.Open()
        self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self._converter = pylon.ImageFormatConverter()
        self._converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self._converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def read(self) -> np.ndarray | None:
        grab = self._camera.RetrieveResult(_RETRIEVE_TIMEOUT_MS, pylon.TimeoutHandling_Return)
        try:
            if not grab.GrabSucceeded():
                return None
            image = self._converter.Convert(grab)
            return image.GetArray()
        finally:
            grab.Release()

    def release(self) -> None:
        self._camera.StopGrabbing()
        self._camera.Close()

    @property
    def node_map(self) -> GenicamNodeMap:
        """Kamera parametrelerini (exposure/gain/trigger/vb.) okuyup yazmak için
        GenICam node-map'e erişim (bkz. `core/camera_params.py`)."""
        return PylonNodeMap(self._camera.GetNodeMap(), camera=self._camera)


GigeCameraSource = BaslerCameraSource
"""Geriye dönük uyumluluk için: sınıf `BaslerCameraSource` olarak yeniden adlandırıldı
(hem GigE hem USB3 Basler kameraları kapsadığı için isim yanıltıcıydı)."""

_RETRIEVE_TIMEOUT_MS = 50
"""`read()` her `_CAMERA_TICK_MS` (main_window.py) periyodunda ana arayüz (GUI) thread'i
üzerinden senkron çağrılır. TriggerMode=On iken tetik gelmezse RetrieveResult verilen süre
kadar BLOKE eder — bu değer eskiden 1000ms'ydi ve tetiksiz beklerken tüm arayüzü saniyede
bir donduruyordu. Kısa bir zaman aşımı, kare gelmediğinde hemen None dönüp bir sonraki
tick'te tekrar denemeyi sağlar; tetik geldiğinde kare hâlâ bu süre içinde yakalanır."""
