"""GenICam node-map erişimini soyutlayan ince arayüz.

pypylon'un ham node tipleri (IInteger/IFloat/IEnumeration/IBoolean) birbirinden farklı
arayüzlere sahiptir; `PylonNodeMap` bunları tek bir `GenicamNode` arayüzüne indirger.
Bu soyutlamanın asıl amacı test edilebilirlik: gerçek pypylon/donanım kurulu olmadan
(bkz. `tests/support/fake_genicam.py`) parametre-kontrol mantığını (`core/camera_params.py`)
sahte bir node-map ile tam kapsamlı test edebilmek.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GenicamNode(Protocol):
    def get_value(self) -> Any: ...

    def set_value(self, value: Any) -> None: ...

    def is_available(self) -> bool: ...

    def is_writable(self) -> bool: ...

    def get_min(self) -> float | None: ...

    def get_max(self) -> float | None: ...

    def get_enum_entries(self) -> list[str] | None: ...


@runtime_checkable
class GenicamNodeMap(Protocol):
    def get_node(self, name: str) -> GenicamNode: ...

    def stop_streaming(self) -> None: ...

    def start_streaming(self) -> None: ...


class PylonNodeMap:
    """Gerçek `camera.GetNodeMap()` (pypylon) üzerine ince bir adaptör.

    `camera` (InstantCamera) opsiyoneldir çünkü node-map bazen kameradan bağımsız test
    edilir; verilirse `stop_streaming`/`start_streaming` gerçek StopGrabbing/StartGrabbing
    çağırır (bkz. `CameraParameterController.set` — Görüntü Formatı alanları GenICam'de
    akış sürerken kilitli olduğu için yazmadan önce akışın durdurulması gerekir).
    """

    def __init__(self, pylon_node_map: Any, camera: Any = None) -> None:
        self._node_map = pylon_node_map
        self._camera = camera

    def get_node(self, name: str) -> GenicamNode:
        return _PylonNode(self._node_map.GetNode(name))

    def stop_streaming(self) -> None:
        if self._camera is not None and self._camera.IsGrabbing():
            self._camera.StopGrabbing()

    def start_streaming(self) -> None:
        if self._camera is not None and not self._camera.IsGrabbing():
            from pypylon import pylon

            self._camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)


class _PylonNode:
    """pypylon'un IInteger/IFloat/IEnumeration/IBoolean node tiplerini sarar.

    pypylon node'ları genelde `GetValue`/`SetValue`/`GetMin`/`GetMax` metodlarını ortak
    isimle sağlar; enum node'lar ek olarak `GetEntries`/`GetSymbolics` sağlar. Bu sınıf
    hangisinin mevcut olduğunu `hasattr` ile kanıtlayıp ona göre davranır — bir node
    tipinin belirli bir metodu olduğunu varsaymaz.
    """

    def __init__(self, node: Any) -> None:
        self._node = node

    def get_value(self) -> Any:
        return self._node.GetValue()

    def set_value(self, value: Any) -> None:
        self._node.SetValue(value)

    def is_available(self) -> bool:
        try:
            from pypylon import genicam

            return bool(genicam.IsAvailable(self._node))
        except ImportError:  # pragma: no cover - pypylon donanım SDK'sı, testte kurulu değil
            return True

    def is_writable(self) -> bool:
        try:
            from pypylon import genicam

            return bool(genicam.IsWritable(self._node))
        except ImportError:  # pragma: no cover - pypylon donanım SDK'sı, testte kurulu değil
            return True

    def get_min(self) -> float | None:
        if not hasattr(self._node, "GetMin"):
            return None
        return self._node.GetMin()

    def get_max(self) -> float | None:
        if not hasattr(self._node, "GetMax"):
            return None
        return self._node.GetMax()

    def get_enum_entries(self) -> list[str] | None:
        if not hasattr(self._node, "GetSymbolics"):
            return None
        return list(self._node.GetSymbolics())
