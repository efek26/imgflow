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

from dataclasses import replace
from typing import Any

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.shape_matching import (
    LabeledMatch,
    build_search_pyramid,
    coarse_search_level,
    find_shape_model,
    render_match_overlay,
)
from imgflow.core.types import PortSpec, PortType
from imgflow.io_utils import shape_model_store
from imgflow.operators import registry


def _parse_model_names(raw: str) -> list[str]:
    return [name.strip() for name in raw.split(",") if name.strip()]


def _apply_search_region(image: Any, params: dict[str, Any]) -> tuple[Any, int, int]:
    """Arama bölgesi tanımlıysa (genişlik VE yükseklik > 0) görüntüyü SADECE arama için
    kırpar ve `(kırpım, ofset_x, ofset_y)` döndürür; tanımlı değilse ya da geçersizse
    `(image, 0, 0)` ile tüm görüntüye düşer (sessizce -- bozuk bir değer pipeline'ı
    çökertmemeli, `parse_roi_list`'in aynı savunmacı deseni)."""
    w = int(params.get("search_w", 0) or 0)
    h = int(params.get("search_h", 0) or 0)
    if w <= 0 or h <= 0:
        return image, 0, 0
    img_h, img_w = image.shape[:2]
    x0 = max(0, min(int(params.get("search_x", 0) or 0), img_w - 1))
    y0 = max(0, min(int(params.get("search_y", 0) or 0), img_h - 1))
    x1 = min(img_w, x0 + w)
    y1 = min(img_h, y0 + h)
    if x1 <= x0 or y1 <= y0:
        return image, 0, 0
    return image[y0:y1, x0:x1], x0, y0


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
            advanced=True,
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
            "angle_step_coarse",
            ParamType.FLOAT,
            default=3.0,
            min=0.5,
            max=15.0,
            step=0.5,
            label="Kaba Arama Açı Adımı",
            help="Kaba (en bulanık) piramit seviyesinde taranan açı örnekleri arasındaki adım "
            "(derece). KÜÇÜLTMEK daha ince/yavaş bir tarama yapar (kaçırma riski daha düşük); "
            "BÜYÜTMEK aramayı HIZLANDIRIR ama iki örnek arasına düşen açılardaki nesneler "
            "kaçırılabilir. 'Şekil bulma kasıyor' durumunda ilk denenecek yer burası DEĞİL — "
            "önce 'Açı Aralığı'nı (hedefin gerçekte dönebileceği aralığa) daraltmayı ve "
            "aramadan önce bir ROI adımıyla görüntüyü küçültmeyi deneyin; bu ikisi genelde "
            "doğruluktan ödün vermeden çok daha büyük hız kazandırır. Varsayılan (3.0) önceki "
            "bir kullanıcı raporunda ('nesne bulunamıyor/kaçırılıyor') kasıtlı seçildi.",
            advanced=True,
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
            advanced=True,
        ),
        ParamSpec(
            "mm_per_px",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            label="Piksel Ölçeği (mm/px)",
            help="Otomatik doldurulur: Lens Kalibrasyonu'nda checkerboard'dan üretilen "
            "deneysel netlik->mesafe modeli varsa HER karede otomatik (elle giriş gerekmez); "
            "yoksa Araçlar > Aktif Yükseklik Ayarla ile eski yöntem kullanılır. 0 ise mm/cm "
            "alanları üretilmez, sadece piksel ölçümleri kullanılır. Doluysa bulunan her "
            "nesnenin genişlik/yüksekliği mm cinsinden (Ölçek Araması açıksa bulunan ölçeğe "
            "göre ayarlanır, kapalıysa öğretilen SABİT boyut), ve öğretildiği pozisyondan x/y "
            "ekseninde ne kadar ötelendiği cm cinsinden sonuçlara eklenir.",
            advanced=True,
        ),
        ParamSpec(
            "scale_min",
            ParamType.FLOAT,
            default=1.0,
            min=0.1,
            max=5.0,
            step=0.05,
            label="Ölçek Min.",
            help="Ölçek Araması: `scale_max` bundan BÜYÜKSE aktif olur, aksi halde (varsayılan, "
            "ikisi de 1.0) hiç ölçek araması yapılmaz -- eskisiyle BİREBİR aynı davranış/hız. "
            "Açıksa öğretilen boyutun bu ORANLA (1.0=aynı boyut, 0.7=%70 küçük) çarpımından "
            "başlayan bir aralık taranır -- gerçek kullanıcı isteği: 'farklı boyutlarda aynı "
            "cisimden varsa scale boyutu yazmalı' (bant yüksekliği/kamera mesafesi "
            "kalibrasyonsuz değiştiğinde ya da fiziksel olarak farklı boyutlu aynı ürün "
            "varyantlarında öğretilen boyuttan sapan nesneleri de bulabilmek için).",
            advanced=True,
        ),
        ParamSpec(
            "scale_max",
            ParamType.FLOAT,
            default=1.0,
            min=0.1,
            max=5.0,
            step=0.05,
            label="Ölçek Maks.",
            help="Ölçek Araması aralığının üst ucu -- `scale_min`'e bakınız. Geniş bir aralık "
            "(ör. 0.5-2.0) aramayı YAVAŞLATIR (her ek ölçek, kaba arama maliyetini aynı oranda "
            "çarpar); sadece gerektiği kadar geniş tutun.",
            advanced=True,
        ),
        ParamSpec(
            "scale_step_coarse",
            ParamType.FLOAT,
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            label="Ölçek Arama Adımı",
            help="Ölçek Araması aktifken kaba taramada `scale_min`-`scale_max` arasında kaç "
            "adımda örnekleneceği. KÜÇÜLTMEK daha hassas ama daha yavaş; BÜYÜTMEK daha hızlı "
            "ama aradaki ölçeklerdeki nesneler kaçırılabilir -- 'Kaba Arama Açı Adımı' ile AYNI "
            "hız/hassasiyet dengesi mantığı.",
            advanced=True,
        ),
        # -- HALCON kökenli sağlamlık/doğruluk/hız parametreleri ------------------------
        ParamSpec(
            "verify_interior",
            ParamType.BOOL,
            default=True,
            label="İç Bölge Doğrulaması",
            help="Skordan BAĞIMSIZ ikinci bir kabul kriteri (HALCON'un clutter/kontrol bölgesi "
            "mantığının karşılığı): eğitimde modelin İÇİ ile çevresi arasındaki kontrast "
            "ölçülür, aramada da aynı ölçüm yapılır ve uymayan adaylar elenir. Gerçek "
            "kullanıcı sorunu: 'Min. Skor'u düşürünce gerçek nesneler bulunuyor AMA düz/boş "
            "zeminde de yanlış eşleşmeler çıkıyordu — bu kriter onları (içi ve dışı aynı "
            "parlaklıkta olduğu için) eler, böylece eşiği güvenle düşürebilirsiniz. Ölçüt "
            "aydınlatmadan bağımsız bir ORAN olduğundan koyu/açık her iki yönde de çalışır. "
            "Bu özellikten ÖNCE eğitilmiş modellerde etkisizdir — modeli yeniden eğitin.",
        ),
        ParamSpec(
            "verify_tolerance",
            ParamType.FLOAT,
            default=0.35,
            min=0.0,
            max=1.0,
            step=0.05,
            label="Doğrulama Toleransı",
            advanced=True,
            help="Bir adayın iç/dış kontrastı, eğitimde ölçülenin en az bu KATI olmalı. "
            "Düşürmek doğrulamayı gevşetir (gölgede kalan gerçek nesneler de geçer), "
            "yükseltmek sıkılaştırır. 0 = sadece kontrast YÖNÜ (koyu/açık) kontrol edilir.",
        ),
        ParamSpec(
            "show_rejected",
            ParamType.BOOL,
            default=False,
            label="Elenen Adayları Göster (teşhis)",
            help="Açıkken eşiğin ALTINDA kalmış ya da doğrulamadan geçememiş en iyi adaylar "
            "görüntü üzerinde KIRMIZI çarpı + skorlarıyla işaretlenir. 'Neden bulamıyor?' "
            "sorusunu tahminle değil görerek yanıtlamak için: gerçek nesnelerin skoru ile "
            "yanlış adayların skoru arasında ayıran bir eşik VAR MI, hemen görürsünüz.",
        ),
        ParamSpec(
            "min_contrast",
            ParamType.FLOAT,
            default=5.0,
            min=0.0,
            max=255.0,
            step=1.0,
            label="Min. Kontrast (gri seviye)",
            help="HALCON'un MinContrast'ının karşılığı: gradyanı bu gri-seviye kontrastın "
            "ALTINDA kalan görüntü pikselleri puanlamaya HİÇ katılmaz. Puanlama sadece kenar "
            "YÖNÜNE baktığından, gradyanı neredeyse sıfır olan düz/gürültülü bir arka plan "
            "pikseli bile rastgele ama tam güçte bir yön üretip skoru kirletiyordu. YÜKSELTMEK "
            "gürültülü/dokulu arka planda yanlış eşleşmeleri azaltır ama ZAYIF kontrastlı "
            "(soluk) gerçek kenarları da eleyebilir — gerçek bir vakada gölgede kalan ürünün "
            "iç kenarları tam da bu yüzden silinip skorunu eşiğin altına düşürebiliyordu, "
            "varsayılan bu nedenle 10 yerine 5. 0 = kapalı.",
            advanced=True,
        ),
        ParamSpec(
            "ignore_polarity",
            ParamType.BOOL,
            default=False,
            label="Polariteyi Yok Say",
            help="HALCON'un Metric='ignore_global_polarity' karşılığı. Kontrast TÜMÜYLE ters "
            "döndüğünde (koyu bant üzerinde açık ürün <-> açık bant üzerinde koyu ürün; ıslak "
            "bant, farklı ürün rengi, arkadan aydınlatma) nesne mükemmel eşleştiği halde skor "
            "-1'e gidip tamamen kaçırılıyordu. Açıkken iki polarite de kabul edilir. Kapalıyken "
            "(varsayılan) ayrım korunur -- ters kontrastlı benzer bir nesneyi AYIRT ETMEK "
            "istiyorsanız kapalı bırakın.",
            advanced=True,
        ),
        ParamSpec(
            "subpixel",
            ParamType.BOOL,
            default=True,
            label="Subpiksel Doğruluk",
            help="HALCON'un SubPixel='interpolation' karşılığı: skor haritasının tepe noktasına "
            "parabol uydurularak konum ve açı, arama ızgarasının adımından daha ince çözülür "
            "(yaklaşık 0.1 px). Maliyeti ihmal edilebilir; öteleme/konum ölçümlerinin "
            "doğruluğunu doğrudan artırır.",
            advanced=True,
        ),
        ParamSpec(
            "last_level",
            ParamType.INT,
            default=0,
            min=0,
            max=5,
            label="Son Piramit Seviyesi (hız)",
            help="İyileştirmenin DURACAĞI piramit seviyesi. 0 (varsayılan) = tam çözünürlüğe "
            "kadar iyileştir (en doğru). 1 vermek iyileştirme maliyetini yaklaşık 4 kat "
            "azaltır (o seviyede dörtte bir piksel var) ama konum/açı hassasiyeti de kabalaşır "
            "-- 'şekil bul çok kasıyor' durumunda ilk denenecek ayar budur. Model bu kadar "
            "seviyeye sahip değilse otomatik olarak sınırlanır.",
            advanced=True,
        ),
        ParamSpec(
            "max_overlap",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=1.0,
            step=0.05,
            label="Maks. Örtüşme",
            help="HALCON'un MaxOverlap karşılığı: iki eşleşmenin sınırlayıcı kutuları bu orandan "
            "FAZLA örtüşüyorsa düşük skorlu olan elenir (oran, KÜÇÜK kutunun alanına göre). "
            "1.0 (varsayılan) = bu kural kapalı, eski mesafe tabanlı bastırma kullanılır. "
            "Bantta YAN YANA/temas eden aynı üründen birden fazla varsa (eski kural onları tek "
            "eşleşmeye indirgeyebiliyordu) 0.3-0.5 civarı bir değer deneyin.",
            advanced=True,
        ),
        ParamSpec(
            "search_x",
            ParamType.INT,
            default=0,
            min=0,
            max=100000,
            label="Arama Bölgesi X",
            help="Aramanın SADECE görüntünün bir bölgesinde yapılmasını sağlar (HALCON'da "
            "aramanın bir Region ile sınırlanmasıyla aynı amaç). 'Arama Bölgesi Genişlik' 0 "
            "iken bu alanlar YOK SAYILIR ve tüm görüntü aranır. Ayrı bir 'ROI / Bölge' adımı "
            "eklemekten farkı: görüntünün kendisi KIRPILMAZ (sonraki adımlar tam kareyi görür) "
            "ve bulunan konumlar tam kare koordinatlarında raporlanır -- sadece arama maliyeti "
            "bölge alanıyla orantılı azalır.",
            advanced=True,
        ),
        ParamSpec("search_y", ParamType.INT, default=0, min=0, max=100000, label="Arama Bölgesi Y",
                  help="Bkz. 'Arama Bölgesi X'.", advanced=True),
        ParamSpec(
            "search_w",
            ParamType.INT,
            default=0,
            min=0,
            max=100000,
            label="Arama Bölgesi Genişlik",
            help="0 (varsayılan) = arama bölgesi KAPALI, tüm görüntü aranır. Bkz. 'Arama Bölgesi X'.",
            advanced=True,
        ),
        ParamSpec("search_h", ParamType.INT, default=0, min=0, max=100000, label="Arama Bölgesi Yükseklik",
                  help="0 = arama bölgesi kapalı. Bkz. 'Arama Bölgesi X'.", advanced=True),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        model_names = _parse_model_names(params.get("model_names") or "")
        if not model_names:
            raise ValueError("'model_names' parametresi boş olamaz (en az bir model seçilmeli).")

        auto_count = bool(params.get("auto_count", True))
        max_matches = None if auto_count else int(params.get("num_matches", 1))
        mm_per_px = float(params.get("mm_per_px", 0.0))
        scale_min = float(params.get("scale_min", 1.0))
        scale_max = float(params.get("scale_max", 1.0))
        scale_search_enabled = scale_max > scale_min
        min_contrast = float(params.get("min_contrast", 10.0))
        max_overlap_param = float(params.get("max_overlap", 1.0))
        common_kwargs = dict(
            angle_start=float(params.get("angle_start", -180.0)),
            angle_extent=float(params.get("angle_extent", 360.0)),
            angle_step_coarse=float(params.get("angle_step_coarse", 3.0)),
            min_score=float(params.get("min_score", 0.7)),
            max_matches=max_matches,
            greediness=float(params.get("greediness", 0.9)),
            scale_min=scale_min,
            scale_max=scale_max,
            scale_step_coarse=float(params.get("scale_step_coarse", 0.1)),
            min_contrast=min_contrast,
            ignore_polarity=bool(params.get("ignore_polarity", False)),
            subpixel=bool(params.get("subpixel", True)),
            last_level=int(params.get("last_level", 0)),
            # 1.0 = "örtüşme kuralı kapalı" (eski mesafe tabanlı bastırma) -- `None` geçilir.
            max_overlap=None if max_overlap_param >= 1.0 else max_overlap_param,
            verify_interior=bool(params.get("verify_interior", True)),
            verify_tolerance=float(params.get("verify_tolerance", 0.35)),
        )
        show_rejected = bool(params.get("show_rejected", False))

        # Arama bölgesi: görüntü KIRPILMAZ (overlay ve sonraki pipeline adımları tam kareyi
        # görür), sadece ARAMA bu kırpımda yapılıp bulunan konumlara bölgenin sol-üst köşesi
        # geri eklenir. `roi.region` adımının aksine bu, koordinat kayması/kalibrasyon
        # yorumunu değiştirmez.
        search_image, offset_x, offset_y = _apply_search_region(image, params)

        # Modeller ÖNCE hep birlikte yüklenir (eski davranışla AYNI sırada, bir isim bulunamazsa
        # AYNI noktada `ShapeModelNotFoundError` fırlar) ki arama görüntüsünün gradyan piramidi
        # TÜM modeller arasında EN DERİN olanı kadar BİR KEZ kurulup paylaşılabilsin -- gerçek
        # kullanıcı raporu: "şekil bulma çok kasıyor" (birden fazla model seçiliyken her model
        # için aynı arama karesinin piramidini sıfırdan yeniden hesaplamak gereksizdi, bkz.
        # `core/shape_matching.py::build_search_pyramid`).
        loaded_models = [(name, shape_model_store.load_shape_model(name)) for name in model_names]
        # Paylaşılan piramit, modellerin GERÇEKTEN kullanacağı en derin seviyeye kadar kurulur
        # (bkz. `coarse_search_level`) -- aday üretiminde atlanan aşırı kaba seviyeleri
        # hesaplamak boşa işti (ölçüldü: 5 seviyeli bir modelde ~85ms).
        last_level_param = int(params.get("last_level", 0))
        max_levels = max(
            (coarse_search_level(model, last_level_param) + 1 for _, model in loaded_models),
            default=0,
        )
        target_pyramid = (
            build_search_pyramid(search_image, max_levels, min_contrast) if max_levels else None
        )

        measurements: list[dict[str, Any]] = []
        entries: list[LabeledMatch] = []
        # Etiket artık model adı+sırası (ör. "cıvata1") DEĞİL, TÜM modeller genelinde tek bir
        # akan sayaç ("1", "2", "3", ...) — gerçek kullanıcı isteği: "her şekili adlandıralım
        # 1,2,3,4 diye adlandırsın". Hangi model olduğu `measurements`'taki ayrı "model" alanında
        # (CSV/hover panelinde) hâlâ mevcut, sadece görüntü üzerindeki/tablodaki numaralama
        # sadeleşti.
        overall_index = 0
        # Teşhis: eşleşme bulunamadığında kullanıcıya "en iyi aday ne kadar yakındı" bilgisi
        # verilir (bkz. `ui/widgets/measurements_summary.py`). Her model için ayrı toplanıp
        # en iyisi raporlanır.
        best_score: float | None = None
        rejected_all: list[dict[str, Any]] = []
        for name, model in loaded_models:
            diagnostics: dict[str, Any] = {}
            matches = find_shape_model(
                search_image, model, target_pyramid=target_pyramid, diagnostics=diagnostics,
                **common_kwargs,
            )
            model_best = diagnostics.get("best_score")
            if model_best is not None and (best_score is None or model_best > best_score):
                best_score = float(model_best)
            for candidate in diagnostics.get("rejected", []):
                shifted = dict(candidate)
                shifted["x"] += offset_x
                shifted["y"] += offset_y
                shifted["model"] = name
                rejected_all.append(shifted)
            if offset_x or offset_y:
                matches = [
                    replace(match, x=match.x + offset_x, y=match.y + offset_y) for match in matches
                ]
            # Öğretilen (ölçek=1.0'daki) sabit genişlik/yükseklik -- ölçek araması AÇIKSA her
            # eşleşmenin GERÇEK boyutu bunun `match.scale` ile çarpımıdır (bkz. aşağısı).
            width_px = float(model.corners[:, 0].max() - model.corners[:, 0].min())
            height_px = float(model.corners[:, 1].max() - model.corners[:, 1].min())
            for match in matches:
                overall_index += 1
                label = str(overall_index)
                measurement: dict[str, Any] = {
                    "model": name,
                    "label": label,
                    "x": match.x,
                    "y": match.y,
                    "angle": match.angle,
                    "score": match.score,
                }
                if scale_search_enabled:
                    # Gerçek kullanıcı isteği: "farklı boyutlarda aynı cisimden varsa scale
                    # boyutu yazmalı" -- SADECE Ölçek Araması açıkken eklenir (kapalıyken
                    # `match.scale` zaten hep 1.0, göstermek gürültü olurdu).
                    measurement["scale_percent"] = match.scale * 100.0
                # Bulunan nesnenin piksel boyutu HER ZAMAN eklenir (kalibrasyon olsun olmasın)
                # -- gerçek kullanıcı isteği: "kalibrasyon seçili olduğu her senaryoda pixelin
                # yanında mm de yazsın veya cm" (önceden kalibrasyon aktifken px DEĞERİ hiç
                # üretilmiyordu, sadece mm gösterilebiliyordu).
                measured_width_px = width_px * match.scale
                measured_height_px = height_px * match.scale
                measurement["width_px"] = measured_width_px
                measurement["height_px"] = measured_height_px
                if mm_per_px > 0:
                    # Gerçek kullanıcı isteği: "geometrik eşlemede scale boyutu ... da
                    # yazmalı" -- bulunan nesnenin GERÇEK DÜNYA boyutu, ölçek araması
                    # açıkken bulunan ölçeğe göre ayarlanır (kapalıyken match.scale=1.0
                    # olduğundan davranış eskisiyle BİREBİR aynı kalır).
                    measurement["width_mm"] = measured_width_px * mm_per_px
                    measurement["height_mm"] = measured_height_px * mm_per_px
                if model.reference_center is not None:
                    # Gerçek kullanıcı isteği: "... cismin ötelenme uzaklığı da yazmalı", ve
                    # sonraki turda: "ötelemeyi x,y şeklinde yaz ve kalibrasyon varsa cm
                    # şeklinde yaz" -- tek bir Öklid mesafesi YÖNÜ gizliyordu (ör. "5mm sağa"
                    # ile "5mm yukarı" aynı skaler değeri üretir); artık işaretli x/y
                    # bileşenleri de ayrı ayrı raporlanıyor. Eski (bu özellikten ÖNCE
                    # eğitilmiş) modellerde `reference_center` `None`'dır -- bu durumda alan
                    # HİÇ eklenmez (yanlış bir "0 öteleme" izlenimi vermemek için).
                    dx = match.x - model.reference_center[0]
                    dy = match.y - model.reference_center[1]
                    displacement_px = (dx * dx + dy * dy) ** 0.5
                    measurement["displacement_px"] = displacement_px
                    measurement["displacement_x_px"] = dx
                    measurement["displacement_y_px"] = dy
                    if mm_per_px > 0:
                        # cm (mm/10) -- gerçek kullanıcı isteği: öteleme mm yerine (daha
                        # büyük/okunması kolay bir birim olan) cm cinsinden yazılsın.
                        measurement["displacement_cm"] = displacement_px * mm_per_px / 10.0
                        measurement["displacement_x_cm"] = dx * mm_per_px / 10.0
                        measurement["displacement_y_cm"] = dy * mm_per_px / 10.0
                measurements.append(measurement)
                entries.append(LabeledMatch(label=label, model=model, match=match))

        rejected_all.sort(key=lambda c: -c["score"])
        overlay = render_match_overlay(image, entries, rejected_all if show_rejected else None)

        if not measurements and best_score is not None:
            # Eşleşme YOKKEN tek bir "teşhis ölçümü" döndürülür: `measurements_summary.py`
            # bunu tanıyıp "en iyi aday skoru X (Min. Skor: Y)" satırını gösterir. Gerçek
            # kullanıcı sorunu: uygulama hiçbir geri bildirim vermediği için eşik körlemesine
            # düşürülüyor, bu da başka yerlerde yanlış eşleşmelere yol açıyordu.
            measurements.append(
                {
                    "no_match": True,
                    "best_score": best_score,
                    "min_score": float(params.get("min_score", 0.7)),
                    "rejected_count": len(rejected_all),
                    "verification_rejected": sum(
                        1 for c in rejected_all if c.get("reason") == "dogrulama"
                    ),
                }
            )
        return {"measurements": measurements, "overlay": overlay}
