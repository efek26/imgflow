"""ONNX (harici eğitilmiş) modellerle nesne tespiti operatörü.

Tek girdili (arama görüntüsü) — `LinearPipeline`'ın checkbox zincirine otomatik
bağlanabilmesi için kasıtlı olarak `geom.shape_match` ile BİREBİR aynı iskelet kullanılır.
Eşleştirilecek MODEL(ler) bu operatörün bir portu DEĞİL; Araçlar > ONNX Model Kaydet...
aracıyla önceden kaydedilip isimle referans alınan `.onnx` dosyalarıdır (bkz.
`io_utils/onnx_model_store.py`) — `model_names` parametresiyle (virgülle ayrılmış, birden
fazla model adı) sadece isimleriyle kullanılır.

Sadece `task_type="yolo"` olarak kaydedilmiş modeller ÇALIŞTIRILABİLİR; "Sınıflandırma"/
"Segmentasyon" türünde kaydedilmiş bir model seçilirse net bir "henüz desteklenmiyor"
hatası verir (crash etmez) — bkz. `io_utils/onnx_model_store.py` modül docstring'i.
"""

from __future__ import annotations

from typing import Any

from imgflow.core.onnx_detection import (
    LabeledDetection,
    OnnxDetectionError,
    create_session,
    find_objects_yolo,
    render_detection_overlay,
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
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        model_names = _parse_model_names(params.get("model_names") or "")
        if not model_names:
            raise ValueError("'model_names' parametresi boş olamaz (en az bir model seçilmeli).")

        min_confidence = float(params.get("min_confidence", 0.25))
        nms_threshold = float(params.get("nms_threshold", 0.45))

        measurements: list[dict[str, Any]] = []
        entries: list[LabeledDetection] = []
        # `shape_match.py`'de kurulan aynı desen: etiket model adı+sırası değil, TÜM
        # modeller genelinde tek bir akan sayaç ("1","2","3",...).
        overall_index = 0
        for name in model_names:
            meta = onnx_model_store.load_model_meta(name)
            if meta["task_type"] != onnx_model_store.TASK_TYPE_YOLO:
                raise OnnxDetectionError(
                    f"'{name}' modelinin türü ('{meta['task_type']}') henüz desteklenmiyor "
                    "— şu an sadece YOLO modelleri çalıştırılabilir."
                )
            session = create_session(meta["path"])
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
                measurements.append(
                    {
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
                )
                entries.append(LabeledDetection(label=label, model_name=name, detection=det))

        overlay = render_detection_overlay(image, entries)
        return {"measurements": measurements, "overlay": overlay}
