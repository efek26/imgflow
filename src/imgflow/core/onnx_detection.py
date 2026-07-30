"""ONNX model çalıştırma — YOLO nesne tespiti, sınıflandırma, semantik segmentasyon; çıktı
çözümleme, NMS, overlay.

`shape_matching.py` ile AYNI ayrım: bu modül operatör/registry'den TAMAMEN bağımsız, saf
algoritma katmanıdır (doğrudan test edilebilir); `operators/builtin/onnx_detect.py` bunun
ince bir ParamSpec/registry sarmalayıcısıdır (task_type'a göre `find_objects_yolo`/
`classify_image`/`segment_image`'e dallanır).

onnxruntime OPSİYONEL bir bağımlılıktır (`pyproject.toml`'daki `ml` extra'sı) —
`core/camera_source.py`'deki pypylon korumasıyla BİREBİR aynı desen: modül seviyesinde
`try/except ImportError`, kullanım anında (`create_session` içinde) net bir Türkçe hata.

**v1 kapsam sınırlamaları (bilinçli sadeleştirmeler, CLAUDE.md'de de not düşülmüştür):**
- YOLO: sadece YOLOv5/v8-tarzı ONNX çıktı biçimleri desteklenir (otomatik ayırt edilir).
- Ön işlemede letterbox/padding YOK, düz `cv2.resize` kullanılır — orijinal görüntü kare
  değilse hafif en/boy oranı bozulması olabilir (üç görev türü de aynı `preprocess`'i paylaşır).
- Sabit RGB + [0,1] normalize varsayılır (mean/std çıkarma YOK) — YOLO ailesinin standart
  ön işleme reçetesi, sınıflandırma/segmentasyon modelleri için de aynen kullanılır.
- Sınıflandırma: tek çıktı tensörü `(1,C)`/`(C,)` beklenir, tek-etiket (multi-label DEĞİL);
  ham çıktıya HER ZAMAN softmax uygulanır (model zaten normalize etmiş olsa bile argmax/
  sıralama değişmez, sadece gösterilen güven değeri etkilenir — bilinçli sadeleştirme).
- Segmentasyon: SADECE semantik (piksel-başına tek sınıf) segmentasyon desteklenir, YOLOv8-seg
  tarzı örnek (instance) segmentasyon (maske katsayıları + prototip maskeler) DEĞİL. Çıktı
  `(1,C,H,W)`/`(1,H,W,C)`/`(C,H,W)` beklenir; `class_id=0` ARKA PLAN varsayılır (overlay'de
  renklendirilmez/ölçüm satırı üretilmez, ama `SegmentationResult.class_areas`'ta raporlanır).
"""

from __future__ import annotations

import ast
import json
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
_OK_COLOR = (0, 255, 0)
_FAIL_COLOR = (0, 0, 255)
"""`color_props.py`'deki `_OK_COLOR`/`_FAIL_COLOR` ile AYNI (BGR) -- `defect_classes`
(bkz. `operators/builtin/onnx_detect.py`) doluyken kutu/metin rengini NG için kırmızıya,
OK için yeşile çevirir; `defect_classes` boşken (tolerance_ok=None) mevcut sabit
`_BOX_COLOR` (yeşil) korunur, davranış değişmez."""

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
    tolerance_ok: bool | None = None
    """`operators/builtin/onnx_detect.py`'nin `defect_classes` parametresi doluyken set
    edilir (tespit edilen sınıf hatalı sınıflardan biriyse False, değilse True) --
    `render_detection_overlay` kutu rengini buna göre seçer. `defect_classes` boşken
    (varsayılan/mevcut davranış) None kalır, renk sabit yeşildir."""


@dataclass(frozen=True)
class ClassificationResult:
    class_id: int
    class_name: str
    confidence: float


@dataclass(frozen=True)
class LabeledClassification:
    label: str
    model_name: str
    result: ClassificationResult
    tolerance_ok: bool | None = None
    """`LabeledDetection.tolerance_ok` ile AYNI anlam/desen -- `defect_classes` doluyken
    tahmin edilen sınıf hatalı sınıflardan biriyse False, değilse True."""


@dataclass(frozen=True)
class SegmentationResult:
    class_map: np.ndarray
    """Girdi görüntüsüyle AYNI (h, w) boyutunda, her piksel için en yüksek skorlu `class_id`
    (int32) -- modelin ürettiği küçük çözünürlükten `cv2.INTER_NEAREST` ile geri ölçeklenir
    (sınıf sınırları LİNEER enterpolasyonla bulanıklaşmasın diye en-yakın-komşu kullanılır)."""
    class_areas: dict[int, int]
    """`class_id -> piksel sayısı`, görüntüde GERÇEKTEN bulunan (en az 1 piksel) TÜM sınıflar
    (0/arka plan DAHİL -- sadece overlay/ölçüm satırı üretiminde `operators/builtin/
    onnx_detect.py` tarafından bilinçli olarak atlanır, bu saf katman hiçbir sınıfı gizlemez)."""


def inspect_onnx_model(model_path: str | Path) -> dict[str, Any]:
    """`{"class_labels": list[str] | None, "input_size": int | None}` döner -- Ultralytics
    YOLO gibi bazı ONNX ihracatlarının modele gömdüğü eğitim metadata'sından (varsa) sınıf
    isimlerini VE modelin girdi tensor şeklinden giriş boyutunu OKUMAYA ÇALIŞIR, kullanıcının
    bunları elle (virgülle) yeniden yazmasına -- ve sırası eşleşmezse sessizce YANLIŞ sınıf
    adı üretmesine (gerçek kullanıcı şikayeti: "sınıflandırmalar bozuk/hatalı gibi geliyor")
    gerek kalmasın diye. `ui/dialogs/onnx_model_dialog.py::_on_choose_file` dosya seçilir
    seçilmez doğrudan çağırır -- bu yüzden ASLA fırlatmaz, okunamayan/eksik/bozuk her durumda
    ilgili alan (ya da ikisi de) sessizce None döner ve kullanıcı elle girer (mevcut/eski
    davranışa düşer)."""
    result: dict[str, Any] = {"class_labels": None, "input_size": None}
    if onnxruntime is None:
        return result
    try:
        session = onnxruntime.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

        raw_names = session.get_modelmeta().custom_metadata_map.get("names")
        if raw_names:
            result["class_labels"] = _parse_class_labels(raw_names)

        shape = session.get_inputs()[0].shape
        if len(shape) == 4:
            height, width = shape[2], shape[3]
            if isinstance(height, int) and isinstance(width, int) and height == width and height > 0:
                result["input_size"] = height
    except Exception:
        # Bozuk/rastgele bir dosya (ör. testteki `_fake_onnx_file`) ya da beklenmeyen bir
        # metadata biçimi -- diyalog dosya seçiminde ÇÖKMEMELİ, sadece elle girişe düşmeli.
        return {"class_labels": result["class_labels"], "input_size": None}
    return result


def _parse_class_labels(raw: str) -> list[str] | None:
    """`raw`, Ultralytics'in yazdığı `"{0: 'kusur', 1: 'saglam'}"` gibi bir Python dict-repr'i
    (JSON DEĞİL -- tek tırnak + tırnaksız int anahtar, `json.loads` bunu parse EDEMEZ) ya da
    bazı ihracatlarda geçerli JSON (`'{"0": "a", "1": "b"}'` ya da `'["a","b"]'`) olabilir."""
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None

    if isinstance(parsed, dict):
        try:
            return [str(parsed[k]) for k in sorted(parsed, key=int)]
        except (TypeError, ValueError):
            return None
    if isinstance(parsed, list):
        return [str(v) for v in parsed]
    return None


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


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def classify_image(
    image: np.ndarray,
    session: Any,
    class_labels: list[str],
    input_size: int = 224,
) -> ClassificationResult:
    if not class_labels:
        raise OnnxDetectionError("Modelin 'class_labels' listesi boş olamaz.")

    blob = preprocess(image, input_size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    arr = np.asarray(outputs[0]).reshape(-1).astype(np.float64)
    if arr.shape[0] != len(class_labels):
        raise OnnxDetectionError(
            f"ONNX çıktı boyutu ({arr.shape[0]}) sınıf sayısıyla ({len(class_labels)}) uyuşmuyor."
        )

    probs = _softmax(arr)
    class_id = int(np.argmax(probs))
    return ClassificationResult(
        class_id=class_id, class_name=class_labels[class_id], confidence=float(probs[class_id])
    )


def _normalize_segmentation_shape(output: np.ndarray, num_classes: int) -> np.ndarray:
    """`(1,C,H,W)`/`(1,H,W,C)`/`(C,H,W)`/`(H,W,C)` girdi çıktı biçimlerini `(C,H,W)`'ye
    normalize eder -- `_normalize_output_shape`'in segmentasyon (2B uzamsal) karşılığı."""
    arr = np.asarray(output)
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise OnnxDetectionError(f"Beklenmeyen segmentasyon çıktı şekli: {output.shape}")
        arr = arr[0]
    if arr.ndim != 3:
        raise OnnxDetectionError(f"Beklenmeyen segmentasyon çıktı şekli: {output.shape}")
    if arr.shape[0] == num_classes:
        return arr
    if arr.shape[-1] == num_classes:
        return np.transpose(arr, (2, 0, 1))
    raise OnnxDetectionError(
        f"Segmentasyon çıktı şekli ({output.shape}) sınıf sayısıyla ({num_classes}) uyuşmuyor."
    )


def segment_image(
    image: np.ndarray,
    session: Any,
    class_labels: list[str],
    input_size: int = 512,
) -> SegmentationResult:
    if not class_labels:
        raise OnnxDetectionError("Modelin 'class_labels' listesi boş olamaz.")

    h, w = image.shape[:2]
    blob = preprocess(image, input_size)
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: blob})
    arr = _normalize_segmentation_shape(np.asarray(outputs[0]), len(class_labels))

    class_map_small = np.argmax(arr, axis=0).astype(np.int32)
    class_map = cv2.resize(class_map_small, (w, h), interpolation=cv2.INTER_NEAREST)
    counts = np.bincount(class_map.reshape(-1), minlength=len(class_labels))
    class_areas = {i: int(counts[i]) for i in range(len(class_labels)) if counts[i] > 0}
    return SegmentationResult(class_map=class_map, class_areas=class_areas)


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
        if entry.tolerance_ok is None:
            color = _BOX_COLOR
            text = f"{entry.label} {d.class_name} %{d.confidence * 100:.0f}"
        else:
            color = _OK_COLOR if entry.tolerance_ok else _FAIL_COLOR
            status = "OK" if entry.tolerance_ok else "NG"
            text = f"{entry.label} {d.class_name} %{d.confidence * 100:.0f} ({status})"
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, box_thickness)
        cv2.putText(
            overlay,
            text,
            (x, max(12, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            text_thickness,
            cv2.LINE_AA,
        )
    return overlay


def render_classification_overlay(
    base_image: np.ndarray, entries: list[LabeledClassification]
) -> np.ndarray:
    """Kutu YOK (sınıflandırma tüm görüntüye ait, tek bir nesneye değil) -- her modelin
    sonucu sol üstte alt alta bir metin satırı olarak yazılır (`render_detection_overlay`
    ile AYNI renk/tolerans deseni: `tolerance_ok is None` -> sabit yeşil, doluysa OK/NG)."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    font_scale = max(0.5, 0.7 * scale_factor)
    text_thickness = max(1, round(2 * scale_factor))
    line_height = max(18, round(26 * scale_factor))

    for i, entry in enumerate(entries):
        r = entry.result
        if entry.tolerance_ok is None:
            color = _BOX_COLOR
            text = f"{entry.label} ({entry.model_name}): {r.class_name} %{r.confidence * 100:.0f}"
        else:
            color = _OK_COLOR if entry.tolerance_ok else _FAIL_COLOR
            status = "OK" if entry.tolerance_ok else "NG"
            text = f"{entry.label} ({entry.model_name}): {r.class_name} %{r.confidence * 100:.0f} ({status})"
        y = line_height * (i + 1)
        cv2.putText(
            overlay, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA
        )
    return overlay


def _class_color(class_id: int) -> tuple[int, int, int]:
    """`class_id`'den deterministik bir BGR renk üretir (sabit doygunluk/parlaklık, sadece
    hue döner) -- harici bir renk paleti kütüphanesi GEREKMEDEN her sınıf tutarlı/tekrar
    üretilebilir bir renkle çizilir."""
    hue = (class_id * 47) % 180
    hsv = np.uint8([[[hue, 200, 255]]])
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


_SEGMENTATION_ALPHA = 0.45


def render_segmentation_overlay(
    base_image: np.ndarray,
    class_map: np.ndarray,
    class_labels: list[str],
    defect_class_ids: set[int] | None = None,
) -> np.ndarray:
    """`class_id=0` ARKA PLAN varsayılır (modül docstring'indeki v1 sınırlama) --
    renklendirilmez ve lejantta gösterilmez. `defect_class_ids` doluysa o sınıfların lejant
    metni kırmızı (`_FAIL_COLOR`), diğerleri kendi `_class_color`'ıyla yazılır."""
    overlay = np.ascontiguousarray(base_image).copy()
    if overlay.ndim == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    defect_class_ids = defect_class_ids or set()

    present = sorted(int(c) for c in np.unique(class_map) if c != 0)
    if present:
        color_layer = np.zeros_like(overlay)
        for class_id in present:
            color_layer[class_map == class_id] = _class_color(class_id)
        mask_any = class_map != 0
        blended = overlay.astype(np.float32)
        blended[mask_any] = (
            overlay[mask_any].astype(np.float32) * (1 - _SEGMENTATION_ALPHA)
            + color_layer[mask_any].astype(np.float32) * _SEGMENTATION_ALPHA
        )
        overlay = blended.astype(np.uint8)

    scale_factor = max(overlay.shape[0], overlay.shape[1]) / _TEXT_SCALE_REFERENCE_DIM
    font_scale = max(0.5, 0.6 * scale_factor)
    text_thickness = max(1, round(1.5 * scale_factor))
    line_height = max(16, round(22 * scale_factor))

    for i, class_id in enumerate(present):
        name = class_labels[class_id] if 0 <= class_id < len(class_labels) else str(class_id)
        color = _FAIL_COLOR if class_id in defect_class_ids else _class_color(class_id)
        y = line_height * (i + 1)
        cv2.putText(
            overlay, name, (10, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, text_thickness, cv2.LINE_AA
        )
    return overlay
