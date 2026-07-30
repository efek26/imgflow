"""ONNX (harici eğitilmiş) modellerle nesne tespiti / sınıflandırma / segmentasyon operatörü.

Tek girdili (arama görüntüsü) — `LinearPipeline`'ın checkbox zincirine otomatik
bağlanabilmesi için kasıtlı olarak `geom.shape_match` ile BİREBİR aynı iskelet kullanılır.
Eşleştirilecek MODEL(ler) bu operatörün bir portu DEĞİL; Araçlar > ONNX Model Kaydet...
aracıyla önceden kaydedilip isimle referans alınan `.onnx` dosyalarıdır (bkz.
`io_utils/onnx_model_store.py`) — `model_names` parametresiyle (virgülle ayrılmış, birden
fazla model adı) sadece isimleriyle kullanılır. Her modelin `task_type`'ı (`"yolo"`/
`"classification"`/`"segmentation"`) kayıt sırasında belirlenmiştir; `run()` bu üçüne göre
`find_objects_yolo`/`classify_image`/`segment_image`'e dallanır — `model_names` listesinde
FARKLI türden modeller AYNI ANDA seçilebilir (overlay'ler üst üste bindirilir, ölçüm
satırları yan yana eklenir).

Opsiyonel `defect_classes` parametresi (virgülle ayrılmış sınıf adları) `analysis.
color_props`/`region_props`'taki `tolerance_enabled` deseniyle AYNI: boşken (varsayılan)
davranış DEĞİŞMEZ; doluyken tespit edilen/tahmin edilen/segmentlenen `class_name` bu
listedeyse `tolerance_ok=False` (NG), değilse `True` (OK) — ilgili `render_*_overlay`
fonksiyonu rengi buna göre kırmızı/yeşile çevirir, `ui/widgets/measurements_summary.py`
OK/NG sayımı gösterir.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from imgflow.core.onnx_detection import (
    LabeledClassification,
    LabeledDetection,
    OnnxDetectionError,
    classify_image,
    create_session,
    find_objects_yolo,
    render_classification_overlay,
    render_detection_overlay,
    render_segmentation_overlay,
    segment_image,
)
from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.io_utils import onnx_model_store
from imgflow.operators import registry


def _parse_model_names(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(",") if name.strip()]


@registry.register
class OnnxDetectOp:
    id = "ml.onnx_detect"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec("measurements", PortType.MEASUREMENTS),
        PortSpec(
            "overlay",
            PortType.IMAGE,
            description="Bulunan nesnelerin kutu + sınıf adı + güven skoruyla işaretlendiği önizleme.",
        ),
    ]
    params = [
        ParamSpec(
            "model_names",
            ParamType.STRING,
            default="",
            label="Model(ler)",
            help="Araçlar > ONNX Model Kaydet... ile kaydedilmiş YOLO model(ler)i. 'Seç...' "
            "ile birden fazlasını işaretleyebilir ya da elle virgülle ayırarak yazabilirsiniz "
            "— aynı görüntüde hepsi birden aranır.",
            dynamic_choices=onnx_model_store.list_models,
            multi_select=True,
        ),
        ParamSpec(
            "min_confidence",
            ParamType.FLOAT,
            default=0.25,
            min=0.0,
            max=1.0,
            step=0.01,
            label="Min. Güven",
            help="Bu eşiğin altındaki tespitler sonuçtan elenir.",
        ),
        ParamSpec(
            "nms_threshold",
            ParamType.FLOAT,
            default=0.45,
            min=0.0,
            max=1.0,
            step=0.01,
            label="NMS Eşiği",
            help="Aynı nesneyi işaret eden örtüşen kutuların tek kutuya indirilme (Non-Max "
            "Suppression) örtüşme eşiği; düşürmek daha agresif birleştirir.",
        ),
        ParamSpec(
            "defect_classes",
            ParamType.STRING,
            default="",
            label="Hatalı Sınıflar",
            help="Opsiyonel: virgülle ayrılmış sınıf adları (ör. 'kusur, cizik'). Bu "
            "listedeki bir sınıf tespit edilirse ölçüm NG (hatalı), diğerleri OK (sağlam) "
            "sayılır ve sonuç panelinde/kutu renginde (kırmızı/yeşil) gösterilir. Boşsa "
            "(varsayılan) tolerans kontrolü kapalı kalır, mevcut davranış değişmez.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        model_names = _parse_model_names(params.get("model_names") or "")
        if not model_names:
            raise ValueError("'model_names' parametresi boş olamaz (en az bir model seçilmeli).")

        min_confidence = float(params.get("min_confidence", 0.25))
        nms_threshold = float(params.get("nms_threshold", 0.45))
        defect_classes = _parse_model_names(params.get("defect_classes") or "")
        defect_set = set(defect_classes)

        measurements: list[dict[str, Any]] = []
        detection_entries: list[LabeledDetection] = []
        classification_entries: list[LabeledClassification] = []
        # `shape_match.py`'de kurulan aynı desen: etiket model adı+sırası değil, TÜM
        # modeller genelinde tek bir akan sayaç ("1","2","3",...).
        overall_index = 0
        # Segmentasyon overlay'i (blend) ARDIŞIK modeller arasında ÜST ÜSTE bindirilir --
        # her `render_*_overlay` çağrısı bir öncekinin sonucunu TABAN alıp üzerine çizer,
        # bu yüzden `image`'in kendisi hiç mutasyona uğramaz (her çağrı kendi kopyasını alır).
        overlay = np.ascontiguousarray(image).copy()

        for name in model_names:
            meta = onnx_model_store.load_model_meta(name)
            task_type = meta["task_type"]
            # Tür doğrulaması modeli YÜKLEMEDEN önce yapılır: bir yapılandırma hatası,
            # onnxruntime'ın çok daha anlaşılmaz "protobuf ayrıştırılamadı" hatasının
            # ardına gizlenmemeli.
            if task_type not in onnx_model_store.SUPPORTED_TASK_TYPES:
                raise OnnxDetectionError(f"'{name}' modelinin türü ('{task_type}') tanınmıyor.")
            session = create_session(meta["path"])

            if task_type == onnx_model_store.TASK_TYPE_YOLO:
                detections = find_objects_yolo(
                    image,
                    session,
                    meta["class_labels"],
                    input_size=meta["input_size"],
                    min_confidence=min_confidence,
                    nms_threshold=nms_threshold,
                )
                for det in detections:
                    overall_index += 1
                    label = str(overall_index)
                    measurement: dict[str, Any] = {
                        "model": name,
                        "label": label,
                        "class_name": det.class_name,
                        "class_id": det.class_id,
                        "confidence": det.confidence,
                        "bbox_x": det.x,
                        "bbox_y": det.y,
                        "bbox_w": det.w,
                        "bbox_h": det.h,
                    }
                    tolerance_ok = None
                    if defect_set:
                        tolerance_ok = det.class_name not in defect_set
                        measurement["tolerance_ok"] = tolerance_ok
                    measurements.append(measurement)
                    detection_entries.append(
                        LabeledDetection(label=label, model_name=name, detection=det, tolerance_ok=tolerance_ok)
                    )

            elif task_type == onnx_model_store.TASK_TYPE_CLASSIFICATION:
                result = classify_image(image, session, meta["class_labels"], input_size=meta["input_size"])
                overall_index += 1
                label = str(overall_index)
                measurement = {
                    "model": name,
                    "label": label,
                    "class_name": result.class_name,
                    "class_id": result.class_id,
                    "confidence": result.confidence,
                }
                tolerance_ok = None
                if defect_set:
                    tolerance_ok = result.class_name not in defect_set
                    measurement["tolerance_ok"] = tolerance_ok
                measurements.append(measurement)
                classification_entries.append(
                    LabeledClassification(label=label, model_name=name, result=result, tolerance_ok=tolerance_ok)
                )

            elif task_type == onnx_model_store.TASK_TYPE_SEGMENTATION:
                seg = segment_image(image, session, meta["class_labels"], input_size=meta["input_size"])
                total_px = image.shape[0] * image.shape[1]
                defect_class_ids: set[int] = set()
                for class_id in sorted(seg.class_areas):
                    if class_id == 0:
                        continue  # class_id=0 arka plan varsayılır (bkz. core/onnx_detection.py)
                    area_px = seg.class_areas[class_id]
                    overall_index += 1
                    label = str(overall_index)
                    labels = meta["class_labels"]
                    class_name = labels[class_id] if 0 <= class_id < len(labels) else str(class_id)
                    measurement = {
                        "model": name,
                        "label": label,
                        "class_id": class_id,
                        "class_name": class_name,
                        "area_px": area_px,
                        "area_percent": area_px / total_px * 100.0,
                    }
                    tolerance_ok = None
                    if defect_set:
                        tolerance_ok = class_name not in defect_set
                        measurement["tolerance_ok"] = tolerance_ok
                        if not tolerance_ok:
                            defect_class_ids.add(class_id)
                    measurements.append(measurement)
                overlay = render_segmentation_overlay(
                    overlay, seg.class_map, meta["class_labels"], defect_class_ids=defect_class_ids
                )

            else:
                raise OnnxDetectionError(f"'{name}' modelinin türü ('{task_type}') tanınmıyor.")

        if detection_entries:
            overlay = render_detection_overlay(overlay, detection_entries)
        if classification_entries:
            overlay = render_classification_overlay(overlay, classification_entries)

        return {"measurements": measurements, "overlay": overlay}
