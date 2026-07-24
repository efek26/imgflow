"""ONNX (YOLO) nesne tespiti — model dosyasını çalıştırma, çıktı çözümleme, NMS, overlay.

`shape_matching.py` ile AYNI ayrım: bu modül operatör/registry'den TAMAMEN bağımsız, saf
algoritma katmanıdır (doğrudan test edilebilir); `operators/builtin/onnx_detect.py` bunun
ince bir ParamSpec/registry sarmalayıcısıdır.

onnxruntime OPSİYONEL bir bağımlılıktır (`pyproject.toml`'daki `ml` extra'sı) —
`core/camera_source.py`'deki pypylon korumasıyla BİREBİR aynı desen: modül seviyesinde
`try/except ImportError`, kullanım anında (`create_session` içinde) net bir Türkçe hata.

**v1 kapsam sınırlamaları (bilinçli sadeleştirmeler, CLAUDE.md'de de not düşülmüştür):**
- Sadece YOLOv5/v8-tarzı ONNX çıktı biçimleri desteklenir (otomatik ayırt edilir).
- Ön işlemede letterbox/padding YOK, düz `cv2.resize` kullanılır — orijinal görüntü kare
  değilse hafif en/boy oranı bozulması olabilir.
- Sabit RGB + [0,1] normalize varsayılır (mean/std çıkarma YOK) — YOLO ailesinin standart
  ön işleme reçetesi.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:  # pragma: no cover - onnxruntime opsiyonel bağımlılık (bkz. 'ml' extra'sı)
    onnxruntime = None

_TEXT_SCALE_REFERENCE_DIM = 1000.0
"""`region_props.py`/`shape_matching.py`/`color_props.py` ile AYNI ölçekleme deseni."""

_BOX_COLOR = (0, 255, 0)

_session_cache: dict[str, Any] = {}
"""Model başına `InferenceSession` inşası pahalıdır (yüzlerce ms) — her kamera tick'inde
(100ms bütçe) yeniden kurmak canlı akışı kilitlerdi, bu yüzden dosya yoluna göre önbelleğe
alınır (aynı oturumda bir model bir kere yüklenir)."""


class OnnxDetectionError(Exception):
    """onnxruntime eksikse, model türü desteklenmiyorsa ya da çıktı şekli çözülemiyorsa fırlatılır."""


@dataclass(frozen=True)
class Detection:
    x: float
    y: float
    w: float
    h: float
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class LabeledDetection:
    label: str
    model_name: str
    detection: Detection


def create_session(model_path: str | Path) -> Any:
    if onnxruntime is None:
        raise OnnxDetectionError(
            "onnxruntime kurulu değil; ONNX desteği için 'pip install imgflow[ml]' "
            "(veya doğrudan 'pip install onnxruntime') gerekir."
        )
    key = str(model_path)
    session = _session_cache.get(key)
    if session is None:
        session = onnxruntime.InferenceSession(key, providers=["CPUExecutionProvider"])
        _session_cache[key] = session
    return session


def preprocess(image: np.ndarray, input_size: int) -> np.ndarray:
    resized = cv2.resize(image, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    rgb = (
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        if resized.ndim == 3
        else cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
    )
    blob = rgb.astype(np.float32) / 255.0
    blob = np.transpose(blob, (2, 0, 1))[None, ...]
    return np.ascontiguousarray(blob)


def _normalize_output_shape(output: np.ndarray, num_classes: int) -> np.ndarray:
    """YOLOv5 (`[1,N,5+C]`) ve YOLOv8 (`[1,4+C,N]`, transpose edilmiş) çıktı biçimlerini
    `(N, 4+C)` ya da `(N, 5+C)` şekline normalize eder -- hangi eksenin "kutu sayısı (N)"
    hangi eksenin "özellik (K)" olduğu `num_classes`'tan türetilen beklenen K değeriyle
    (4+C ya da 5+C) eşleştirilerek ayırt edilir."""
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise OnnxDetectionError(f"Beklenmeyen ONNX çıktı şekli: {output.shape}")

    k4, k5 = 4 + num_classes, 5 + num_classes
    rows, cols = arr.shape
    if cols in (k4, k5):
        return arr
    if rows in (k4, k5):
        return arr.T
    raise OnnxDetectionError(
        f"ONNX çıktı şekli ({output.shape}) 'class_labels' sayısıyla ({num_classes}) uyuşmuyor."
    )


def _decode_boxes(
    arr: np.ndarray, num_classes: int, min_confidence: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cols = arr.shape[1]
    has_objectness = cols == 5 + num_classes

    cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if has_objectness:
        objectness = arr[:, 4]
        class_scores = arr[:, 5 : 5 + num_classes]
        confidence = objectness * class_scores.max(axis=1)
    else:
        class_scores = arr[:, 4 : 4 + num_classes]
        confidence = class_scores.max(axis=1)
    class_ids = class_scores.argmax(axis=1)

    keep = confidence >= min_confidence
    return cx[keep], cy[keep], w[keep], h[keep], class_ids[keep], confidence[keep]


def find_objects_yolo(
    image: np.ndarray,
    session: Any,
    class_labels: list[str],
    input_size: int = 640,
    min_confidence: float = 0.25,
    nms_threshold: float = 0.45,
) -> list[Detection]:
    if not class_labels:
        raise OnnxDetectionError("Modelin 'class_labels' listesi boş olamaz.")

    h, w = image.shape[:2]
    blob = preprocess(image, input_size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    arr = _normalize_output_shape(outputs[0], len(class_labels))
    cx, cy, bw, bh, class_ids, confidence = _decode_boxes(arr, len(class_labels), min_confidence)
    if cx.size == 0:
        return []

    x_scale, y_scale = w / input_size, h / input_size
    x1 = (cx - bw / 2.0) * x_scale
    y1 = (cy - bh / 2.0) * y_scale
    box_w = bw * x_scale
    box_h = bh * y_scale
    boxes = np.stack([x1, y1, box_w, box_h], axis=1)

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), confidence.tolist(), min_confidence, nms_threshold)
    if len(indices) == 0:
        return []
    indices = np.asarray(indices).reshape(-1)

    detections = []
    for idx in indices:
        class_id = int(class_ids[idx])
        class_name = class_labels[class_id] if 0 <= class_id < len(class_labels) else str(class_id)
        detections.append(
            Detection(
                x=float(boxes[idx, 0]),
                y=float(boxes[idx, 1]),
                w=float(boxes[idx, 2]),
                h=float(boxes[idx, 3]),
                class_id=class_id,
                class_name=class_name,
                confidence=float(confidence[idx]),
            )
        )
    return detections


def render_detection_overlay(base_image: np.ndarray, entries: list[LabeledDetection]) -> np.ndarray:
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    box_thickness = max(2, round(2 * scale_factor))
    font_scale = max(0.5, 0.6 * scale_factor)
    text_thickness = max(1, round(1.5 * scale_factor))

    for entry in entries:
        d = entry.detection
        x, y, w, h = int(round(d.x)), int(round(d.y)), int(round(d.w)), int(round(d.h))
        cv2.rectangle(overlay, (x, y), (x + w, y + h), _BOX_COLOR, box_thickness)
        text = f"{entry.label} {d.class_name} %{d.confidence * 100:.0f}"
        cv2.putText(
            overlay,
            text,
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            _BOX_COLOR,
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay
