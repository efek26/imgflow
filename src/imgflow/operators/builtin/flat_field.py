"""Aydınlatma/vignetting düzeltme (flat-field correction) operatörü.

İKİ mod destekler (`mode` parametresi):

- **"reference" (varsayılan)**: HALCON'daki `div_image` mantığına benzer -- kullanıcının
  önceden Araçlar > Aydınlatma Referansı Kaydet... ile kaydettiği BOŞ/düz aydınlatılmış bir
  referans karenin gri versiyonundan bir "kazanç haritası" (`gain = ortalama(ref) / ref`)
  çıkarılır; her yeni karede bu kazanç uygulanarak (`corrected = image * gain`) kenarlara
  doğru kararan (vignetting) ya da eşit olmayan SABİT aydınlatma düzeltilir. Referans,
  isimli-profil deseniyle (`io_utils/flatfield_store.py`) sadece isimle (`reference_name`)
  referans alınır. **Bilinçli sınırlama:** referans BOŞ bir kareden alındığı için ürünün
  KENDİ gölgesi referansta hiç YOKTUR -- bu mod sabit/zemine ait aydınlatma sorunlarını
  düzeltir ama ürün gölgelerine DOKUNMAZ (gerçek kullanıcı raporu: "gölgeleri yok etmiyor,
  sadece dışarıdaki alanı düzeltiyor" -- referans-tabanlı yöntemin doğası gereği beklenen
  bir davranış, bug değil).
- **"dynamic_local" (yeni)**: referans GEREKTİRMEZ -- her karenin KENDİSİNİN büyük ölçekli
  bulanık halini (`cv2.GaussianBlur`, `blur_radius`) o an için "yerel arka plan" tahmini
  olarak kullanır ve AYNI `gain = ortalama/arka_plan` formülüyle böler. Bu tahmin HER karede
  YENİDEN hesaplandığından, o karede o an bulunan ürünün gölgesini de (gölge tipik olarak
  ürünün kendi ince detaylarından daha YAVAŞ/geniş bir parlaklık değişimidir, bu yüzden
  yeterince büyük bir bulanıklık yarıçapı onu "arka plan"ın bir parçası sayar) kapsar --
  "reference" modunun YAPISAL OLARAK yapamadığı şeyi yapar. Bu, klasik "rolling-ball"/yerel
  aydınlatma düzleştirme tekniğinin (mikroskopi/OCR ön-işlemede yaygın) basitleştirilmiş bir
  hâlidir.

İkisi de `max_gain`/`strength` güvenlik ağını PAYLAŞIR (bkz. aşağıdaki `run()`).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.io_utils import flatfield_store
from imgflow.operators import registry

_EPS = 1e-3


@registry.register
class FlatFieldOp:
    id = "correction.flat_field"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "mode",
            ParamType.ENUM,
            default="reference",
            choices=["reference", "dynamic_local"],
            label="Mod",
            help="'Referans': önceden kaydedilmiş BOŞ bir kareye göre SABİT vignetting/eşit "
            "olmayan aydınlatmayı düzeltir -- ürünün kendi gölgesi referansta YOKTUR, bu "
            "yüzden ürün gölgelerini düzeltMEZ, sadece zeminin/optiğin sabit karanlığını "
            "düzeltir. 'Yerel/Dinamik': referans GEREKTİRMEZ, HER karenin kendi büyük ölçekli "
            "bulanık halini o anki yerel arka plan tahmini olarak kullanır -- bu, o karede o "
            "an var olan ÜRÜN GÖLGELERİNİ de kapsar. 'Bulanıklık Yarıçapı' ürünün kendi "
            "detaylarından BÜYÜK ama gölge/aydınlatma değişiminden KÜÇÜK seçilmeli.",
        ),
        ParamSpec(
            "reference_name",
            ParamType.STRING,
            default="",
            label="Aydınlatma Referansı (sadece 'Referans' modunda)",
            help="Araçlar > Aydınlatma Referansı Kaydet... ile önceden kaydedilmiş BOŞ/düz "
            "aydınlatılmış bir referans kare (ör. boş bant). Bu referansa göre kenarlara "
            "doğru kararan (vignetting) ya da eşit olmayan aydınlatma düzeltilir. Mod "
            "'Yerel/Dinamik' iken bu alan YOK SAYILIR.",
            dynamic_choices=flatfield_store.list_references,
        ),
        ParamSpec(
            "blur_radius",
            ParamType.INT,
            default=31,
            min=3,
            max=301,
            step=2,
            label="Bulanıklık Yarıçapı (px, sadece 'Yerel/Dinamik' modda)",
            help="Yerel arka plan tahmini için kullanılan Gauss bulanıklığının yarıçapı. ÇOK "
            "KÜÇÜKSE ürünün kendi kenarları/detayları da 'arka plan' sayılıp silinir (ürün "
            "hayalet gibi görünür); ÇOK BÜYÜKSE gölge/aydınlatma değişimi yeterince "
            "düzleştirilemez ve düzeltilmeden kalır. Ürünün piksel boyutundan büyük bir "
            "değerle başlayıp canlı önizlemeye bakarak ayarlayın. Tek sayıya yuvarlanır "
            "(Gauss çekirdeği tek boyutlu olmalı).",
        ),
        ParamSpec(
            "strength",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=1.0,
            step=0.05,
            label="Düzeltme Gücü",
            help="0 = düzeltme uygulanmaz (orijinal görüntü), 1 = tam düzeltme. Aşırı "
            "düzeltmenin gürültüyü de büyüttüğü durumlarda düşürülebilir.",
        ),
        ParamSpec(
            "max_gain",
            ParamType.FLOAT,
            default=3.0,
            min=1.0,
            max=20.0,
            step=0.5,
            label="Maks. Kazanç",
            help="Referans karede KOYU/gölgeli bir bölge varsa (ör. bandın dışına taşan "
            "kenar, eşit olmayan aydınlatma) kazanç haritası (gain = ortalama/referans) o "
            "bölgede sınırsız büyüyüp görüntünün o kısmını -hatta referans yeterince genel "
            "koyuysa görüntünün büyük bir kısmını- tamamen BEYAZA (255) doyurabilir; ekranda "
            "'görüntü kayboldu' gibi görünür. Bu değer kazancın çarpanını sınırlar (ör. 3 = "
            "bir piksel en fazla 3 kat parlatılır); referans daha DÜZGÜN/az kararan bir "
            "kareyle yeniden kaydedilene kadar geçici bir güvenlik ağıdır.",
        ),
        ParamSpec(
            "mult",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=10.0,
            step=0.1,
            label="Çarpan (Mult)",
            help="HALCON'un div_image(Image1, Image2, Mult, Add) operatöründeki Mult ile AYNI: "
            "düzeltilmiş sonucu ek bir sabit çarpanla ölçekler. Düzeltme sonrası görüntü genel "
            "olarak çok karanlık/çok parlak kalıyorsa kullanılır. 1.0 = değişiklik yok.",
        ),
        ParamSpec(
            "add",
            ParamType.FLOAT,
            default=0.0,
            min=-255.0,
            max=255.0,
            step=1.0,
            label="Ekleme (Add)",
            help="HALCON'un div_image(...) operatöründeki Add ile AYNI: düzeltilmiş sonuca "
            "sabit bir parlaklık ofseti ekler. 0 = değişiklik yok.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        mode = params.get("mode", "reference")
        strength = float(params.get("strength", 1.0))
        max_gain = float(params.get("max_gain", 3.0))

        if mode == "dynamic_local":
            gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur_radius = int(params.get("blur_radius", 31))
            if blur_radius % 2 == 0:
                blur_radius += 1
            # `sigmaX=0` -> OpenCV çekirdek boyutundan (blur_radius) otomatik sigma türetir --
            # `strength`'teki gibi kullanıcı ek bir sigma parametresiyle uğraşmaz.
            bg_gray = cv2.GaussianBlur(gray.astype(np.float32), (blur_radius, blur_radius), 0)
            gain_reference_mean = float(gray.mean())
        else:
            reference_name = (params.get("reference_name") or "").strip()
            if not reference_name:
                raise ValueError(
                    "'reference_name' parametresi boş olamaz (önce Araçlar > Aydınlatma "
                    "Referansı Kaydet... ile bir referans kaydedin, ya da Mod'u "
                    "'Yerel/Dinamik' olarak değiştirin)."
                )
            reference = flatfield_store.load_reference(reference_name)
            bg_gray = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
            if bg_gray.shape != image.shape[:2]:
                bg_gray = cv2.resize(
                    bg_gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
                )
            gain_reference_mean = float(bg_gray.mean())

        # Kazanç haritası: "arka plan" tahmininin (statik referans YA DA o karenin kendi yerel
        # bulanık hali) ortalamasına göre HER pikselin ne kadar karartılmış olduğunu verir --
        # bu oranla çarpmak aynı düzensizliği taşıyan görüntüde aynı düzeltmeyi uygular. Arka
        # planın KOYU bölgelerinde (-> 0) bu oran sınırsız büyür -- `max_gain` ile
        # sınırlanmazsa o bölgeler (ve arka plan genel olarak yeterince koyuysa görüntünün
        # büyük bir kısmı) BEYAZA doyup ekranda görüntü kaybolmuş gibi görünür.
        gain = gain_reference_mean / (bg_gray.astype(np.float32) + _EPS)
        gain = np.clip(gain, 0.0, max_gain)
        if image.ndim == 3:
            gain = gain[:, :, None]

        working = image.astype(np.float32)
        corrected = working * gain
        if strength < 1.0:
            corrected = working * (1.0 - strength) + corrected * strength

        # HALCON'un div_image(Image1, Image2, Mult, Add) çıktı ölçeklemesiyle AYNI; varsayılan
        # (1.0 / 0.0) değerlerde sonuç bu adım hiç yokmuş gibi kalır.
        mult = float(params.get("mult", 1.0))
        add = float(params.get("add", 0.0))
        if mult != 1.0 or add != 0.0:
            corrected = corrected * mult + add

        return {"image": np.clip(corrected, 0, 255).astype(np.uint8)}
