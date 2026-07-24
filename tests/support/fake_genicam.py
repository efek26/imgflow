"""Gerçek pypylon/donanım olmadan GenICam node-map mantığını test etmek için sahte double'lar."""

from __future__ import annotations

from typing import Any


class FakeGenicamNode:
    def __init__(
        self,
        value: Any,
        min: float | None = None,  # noqa: A002 - GenICam terminolojisiyle tutarlı isim
        max: float | None = None,  # noqa: A002
        available: bool = True,
        writable: bool = True,
        enum_entries: list[str] | None = None,
    ) -> None:
        self._value = value
        self._min = min
        self._max = max
        self._available = available
        self._writable = writable
        self._enum_entries = enum_entries

    def get_value(self) -> Any:
        return self._value

    def set_value(self, value: Any) -> None:
        if not self._writable:
            raise RuntimeError("Node yazılabilir değil.")
        if self._min is not None:
            value = max(value, self._min)
        if self._max is not None:
            value = min(value, self._max)
        self._value = value

    def is_available(self) -> bool:
        return self._available

    def is_writable(self) -> bool:
        return self._writable

    def get_min(self) -> float | None:
        return self._min

    def get_max(self) -> float | None:
        return self._max

    def get_enum_entries(self) -> list[str] | None:
        return self._enum_entries


class FakeNodeMap:
    """`GenicamNodeMap` sözleşmesini sahte node'lardan oluşan bir sözlükle karşılar.

    Bilinmeyen bir node adı istenirse KeyError fırlatır (sessizce None/varsayılan
    döndürmez) — CLAUDE.md'nin "kanıtla, tahmin etme" ilkesiyle tutarlı.
    """

    def __init__(self, nodes: dict[str, FakeGenicamNode]) -> None:
        self._nodes = nodes
        self.stop_streaming_calls = 0
        self.start_streaming_calls = 0

    def get_node(self, name: str) -> FakeGenicamNode:
        return self._nodes[name]

    def stop_streaming(self) -> None:
        self.stop_streaming_calls += 1

    def start_streaming(self) -> None:
        self.start_streaming_calls += 1


def default_camera_nodes(**overrides: FakeGenicamNode) -> dict[str, FakeGenicamNode]:
    """`core/camera_params.py`'deki tüm katalog node'ları için makul varsayılan değerler.

    `CameraParameterController.build_specs()` katalogdaki HER node'u sorguladığı için
    (mevcut/yazılabilir olmasa da), eksiksiz bir sahte node-map burada tek yerden üretilir.
    """
    nodes = {
        "ExposureTime": FakeGenicamNode(1000.0, min=10.0, max=10000.0),
        "Gain": FakeGenicamNode(0.0, min=0.0, max=24.0),
        "BlackLevel": FakeGenicamNode(0.0, min=0.0, max=64.0),
        "BalanceRatioSelector": FakeGenicamNode("Red", enum_entries=["Red", "Green", "Blue"]),
        "BalanceRatio": FakeGenicamNode(1.0, min=0.0, max=4.0),
        "PixelFormat": FakeGenicamNode("Mono8", enum_entries=["Mono8", "RGB8"]),
        "Width": FakeGenicamNode(1920, min=1, max=1920),
        "Height": FakeGenicamNode(1080, min=1, max=1080),
        "OffsetX": FakeGenicamNode(0, min=0, max=1920),
        "OffsetY": FakeGenicamNode(0, min=0, max=1080),
        "BinningHorizontal": FakeGenicamNode(1, min=1, max=4),
        "BinningVertical": FakeGenicamNode(1, min=1, max=4),
        "DecimationHorizontal": FakeGenicamNode(1, min=1, max=4),
        "DecimationVertical": FakeGenicamNode(1, min=1, max=4),
        "ReverseX": FakeGenicamNode(False),
        "ReverseY": FakeGenicamNode(False),
        "AcquisitionFrameRate": FakeGenicamNode(30.0, min=1.0, max=120.0),
        "TriggerMode": FakeGenicamNode("Off", enum_entries=["Off", "On"]),
        "TriggerSource": FakeGenicamNode(
            "Software", enum_entries=["Software", "Line1", "Line2"], available=False
        ),
        "TriggerActivation": FakeGenicamNode(
            "RisingEdge", enum_entries=["RisingEdge", "FallingEdge"]
        ),
        "LineSelector": FakeGenicamNode("Line1", enum_entries=["Line1", "Line2"]),
        "LineMode": FakeGenicamNode("Input", enum_entries=["Input", "Output"]),
        "LineSource": FakeGenicamNode("ExposureActive", enum_entries=["ExposureActive"]),
        "GevSCPSPacketSize": FakeGenicamNode(1500, min=220, max=9000, available=False),
        "GevSCPD": FakeGenicamNode(0, min=0, max=100000, available=False),
        "DeviceLinkThroughputLimit": FakeGenicamNode(
            125000000, min=1000000, max=125000000, available=False
        ),
        "UserSetSelector": FakeGenicamNode(
            "Default", enum_entries=["Default", "UserSet1", "UserSet2"]
        ),
        "UserSetSave": FakeGenicamNode(False),
        "UserSetDefault": FakeGenicamNode(
            "Default", enum_entries=["Default", "UserSet1", "UserSet2"]
        ),
    }
    nodes.update(overrides)
    return nodes
