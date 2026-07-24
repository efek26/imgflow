"""HALCON-tarzı geometrik (şekil) eşleştirme operatörü.

Tek girdili (arama görüntüsü) — `LinearPipeline`'ın checkbox zincirine otomatik
bağlanabilmesi için kasıtlı olarak böyle tasarlandı (bkz. CLAUDE.md). Eşleştirilecek MODEL(ler)
bu operatörün bir portu DEĞİL; Araçlar > Şekil Eşleştirme (Model Öğret) aracıyla önceden
eğitilip isimle kaydedilmiş `core/shape_matching.ShapeModel`lerdir (bkz.
`io_utils/shape_model_store.py`) — `model_names` parametresiyle (virgülle ayrılmış, birden
fazla model adı) sadece isimleriyle referans alınır (Lens/Yükseklik kalibrasyon profillerinin
isimle referans alınmasıyla aynı desen).

Aynı adımda BİRDEN FAZLA model (ör. hem "cıvata" hem "somun") aranabilir: her model adı için
`find_shape_model` ayrı ayrı çalıştırılır (aynı arama görüntüsü üzerinde), sonuçlar TEK bir
`measurements` listesinde birleştirilir ve her eşleşme `f"{model_adı}{sıra}"` (ör. "cıvata1",
"cıvata2") şeklinde MODEL BAZINDA sıra numarasıyla etiketlenir.
"""

from __future__ import annotations

from typing import Any

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.shape_matching import LabeledMatch, find_shape_model, render_match_overlay
from imgflow.core.types import PortSpec, PortType
from imgflow.io_utils import shape_model_store
from imgflow.operators import registry


def _parse_model_names(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(",") if name.strip()]


@registry.register
class ShapeMatchOp:
    id = "geom.shape_match"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec("measurements", PortType.MEASUREMENTS),
        PortSpec(
            "overlay",
            PortType.IMAGE,
            description="Bulunan model(ler)in pozunu (dikdörtgen kontur + açı ekseni + etiket) gösteren önizleme.",
        ),
    ]
    params = [
        ParamSpec(
            "model_names",
            ParamType.STRING,
            default="",
            label="Model(ler)",
            help="Araçlar > Şekil Eşleştirme (Model Öğret) ile eğitilip kaydedilmiş model(ler). "
            "'Seç...' ile kayıtlı modellerden birden fazlasını işaretleyebilir ya da elle "
            "virgülle ayırarak yazabilirsiniz (ör. 'cıvata, somun') — aynı görüntüde hepsi "
            "birden aranır.",
            dynamic_choices=shape_model_store.list_shape_models,
            multi_select=True,
        ),
        ParamSpec(
            "angle_start",
            ParamType.FLOAT,
            default=-180.0,
            min=-360.0,
            max=360.0,
            label="Açı Başlangıç",
            help="Aramanın başlayacağı açı (derece). Modelin hedefte HANGİ açı aralığında "
            "dönmüş olabileceğini kısıtlamak arama hızını da artırır.",
        ),
        ParamSpec(
            "angle_extent",
            ParamType.FLOAT,
            default=360.0,
            min=0.0,
            max=360.0,
            label="Açı Aralığı",
            help="'Açı Başlangıç'tan itibaren kaç derecelik bir yelpazenin taranacağı. 360 = tüm "
            "yönler; ürün hattı üzerinde sabit/bilinen bir yönelimdeyse daraltmak hem yanlış "
            "eşleşmeleri azaltır hem de aramayı hızlandırır.",
        ),
        ParamSpec(
            "min_score",
            ParamType.FLOAT,
            default=0.7,
            min=0.0,
            max=1.0,
            step=0.01,
            label="Min. Skor",
            help="0-1 arası kabul eşiği (1 = modelle birebir kenar-yön uyumu). Bu değerin altındaki "
            "eşleşmeler sonuçtan elenir; aydınlatma/kontrast değişimi çoksa düşürün, yanlış "
            "pozitifleri azaltmak için yükseltin.",
        ),
        ParamSpec(
            "auto_count",
            ParamType.BOOL,
            default=True,
            label="Otomatik (Bul: Hepsi)",
            help="Açıkken 'Min. Skor' eşiğini geçen TÜM örnekler bulunur (kaç tane olduğu "
            "bilinmiyorsa/değişkense). Kapatıp 'Eşleşme Sayısı' ile elle bir üst sınır "
            "belirleyebilirsiniz (ör. hatta her zaman tam olarak 4 parça olduğunu biliyorsanız).",
        ),
        ParamSpec(
            "num_matches",
            ParamType.INT,
            default=1,
            min=1,
            max=100,
            label="Eşleşme Sayısı (manuel)",
            help="'Otomatik' KAPALIYKEN kullanılır: model başına en fazla kaç örnek döndürüleceği.",
        ),
        ParamSpec(
            "greediness",
            ParamType.FLOAT,
            default=0.9,
            min=0.01,
            max=1.0,
            step=0.01,
            label="Açgözlülük",
            help="Düşük değer, kaba piramit seviyesinde daha gevşek bir ön-eleme eşiği kullanarak "
            "arama hızını artırır ama yanlış negatif riskini de artırır.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        model_names = _parse_model_names(params.get("model_names") or "")
        if not model_names:
            raise ValueError("'model_names' parametresi boş olamaz (en az bir model seçilmeli).")

        auto_count = bool(params.get("auto_count", True))
        max_matches = None if auto_count else int(params.get("num_matches", 1))
        common_kwargs = dict(
            angle_start=float(params.get("angle_start", -180.0)),
            angle_extent=float(params.get("angle_extent", 360.0)),
            min_score=float(params.get("min_score", 0.7)),
            max_matches=max_matches,
            greediness=float(params.get("greediness", 0.9)),
        )

        measurements: list[dict[str, Any]] = []
        entries: list[LabeledMatch] = []
        # Etiket artık model adı+sırası (ör. "cıvata1") DEĞİL, TÜM modeller genelinde tek bir
        # akan sayaç ("1", "2", "3", ...) — gerçek kullanıcı isteği: "her şekili adlandıralım
        # 1,2,3,4 diye adlandırsın". Hangi model olduğu `measurements`'taki ayrı "model" alanında
        # (CSV/hover panelinde) hâlâ mevcut, sadece görüntü üzerindeki/tablodaki numaralama
        # sadeleşti.
        overall_index = 0
        for name in model_names:
            model = shape_model_store.load_shape_model(name)
            matches = find_shape_model(image, model, **common_kwargs)
            for match in matches:
                overall_index += 1
                label = str(overall_index)
                measurements.append(
                    {"model": name, "label": label, "x": match.x, "y": match.y, "angle": match.angle, "score": match.score}
                )
                entries.append(LabeledMatch(label=label, model=model, match=match))

        overlay = render_match_overlay(image, entries)
        return {"measurements": measurements, "overlay": overlay}
