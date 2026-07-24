"""Aydınlatma/vignetting düzeltme (flat-field correction) operatörü.

HALCON'daki `div_image` mantığına benzer: kullanıcının önceden Araçlar > Aydınlatma
Referansı Kaydet... ile kaydettiği BOŞ/düz aydınlatılmış bir referans karenin gri
versiyonundan bir "kazanç haritası" (`gain = ortalama(ref) / ref`) çıkarılır; her yeni
karede bu kazanç uygulanarak (`corrected = image * gain`) kenarlara doğru kararan
(vignetting) ya da eşit olmayan aydınlatma düzeltilir. Referans, isimli-profil deseniyle
(`io_utils/flatfield_store.py`) sadece isimle (`reference_name`) referans alınır — ikinci
bir zorunlu girdiye ihtiyaç duymadan `LinearPipeline`'ın tek-girdi/tek-çıktı checkbox
zincirine oturur (bkz. CLAUDE.md).
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
            "reference_name",
            ParamType.STRING,
            default="",
            label="Aydınlatma Referansı",
            help="Araçlar > Aydınlatma Referansı Kaydet... ile önceden kaydedilmiş BOŞ/düz "
            "aydınlatılmış bir referans kare (ör. boş bant). Bu referansa göre kenarlara "
            "doğru kararan (vignetting) ya da eşit olmayan aydınlatma düzeltilir.",
            dynamic_choices=flatfield_store.list_references,
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
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        reference_name = (params.get("reference_name") or "").strip()
        if not reference_name:
            raise ValueError(
                "'reference_name' parametresi boş olamaz (önce Araçlar > Aydınlatma "
                "Referansı Kaydet... ile bir referans kaydedin)."
            )

        reference = flatfield_store.load_reference(reference_name)
        strength = float(params.get("strength", 1.0))

        ref_gray = reference if reference.ndim == 2 else cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
        if ref_gray.shape != image.shape[:2]:
            ref_gray = cv2.resize(
                ref_gray, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR
            )

        # Kazanç haritası: referansın ortalamasına göre HER pikselin ne kadar karartılmış
        # olduğunu (vignetting oranını) verir -- bu oranla çarpmak, aynı optik/aydınlatma
        # düzensizliğini taşıyan HERHANGİ bir görüntüde aynı düzeltmeyi uygular.
        gain = float(ref_gray.mean()) / (ref_gray.astype(np.float32) + _EPS)
        if image.ndim == 3:
            gain = gain[:, :, None]

        working = image.astype(np.float32)
        corrected = working * gain
        if strength < 1.0:
            corrected = working * (1.0 - strength) + corrected * strength

        return {"image": np.clip(corrected, 0, 255).astype(np.uint8)}
