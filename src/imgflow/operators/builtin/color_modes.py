"""Renk uzayına özgü operatörler: Siyah-Beyaz, RGB, HSV.

gorsel_isleme_platformu (önceki Tkinter prototip) referans alınmıştır: tek bir "renk
dönüşümü" operatörü + koşullu görünür alanlar yerine, her mod kendi operatörüdür — bu
sayede UI'da biri işaretlendiğinde ParamForm zaten sadece o operatörün (dolayısıyla o
moda özgü) parametrelerini gösterir, ayrı bir "koşullu görünürlük" mekanizması gerekmez.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


def _ensure_odd(k: Any) -> int:
    k = max(1, int(k))
    return k if k % 2 == 1 else k + 1


def _gray_world_white_balance(img_bgr: np.ndarray) -> np.ndarray:
    """Sahnenin ortalama renginin nötr gri olması varsayımıyla kanalları ayrı ölçekler."""
    img = img_bgr.astype(np.float32)
    b, g, r = cv2.split(img)
    b_avg, g_avg, r_avg = b.mean(), g.mean(), r.mean()
    gray_avg = (b_avg + g_avg + r_avg) / 3.0
    b *= gray_avg / (b_avg + 1e-6)
    g *= gray_avg / (g_avg + 1e-6)
    r *= gray_avg / (r_avg + 1e-6)
    return np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)


@registry.register
class GrayscaleOp:
    id = "color.grayscale"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "clahe_enabled",
            ParamType.BOOL,
            default=False,
            label="CLAHE (Bölgesel Kontrast)",
            help="Görüntünün karanlık/aydınlık bölgelerini ayrı ayrı dengeler; homojen olmayan aydınlatmada nesneyi belirginleştirir.",
        ),
        ParamSpec(
            "threshold_enabled",
            ParamType.BOOL,
            default=False,
            label="Eşikleme",
            help="Görüntüyü siyah/beyaz (0/255) ikili görüntüye çevirir; nesne/arka plan ayrımı ve alan ölçümü için ilk adımdır.",
        ),
        ParamSpec(
            "threshold_mode",
            ParamType.ENUM,
            default="OTSU",
            choices=["OTSU", "ADAPTIVE", "MANUAL"],
            label="Eşikleme Modu",
            help="OTSU: eşiği otomatik bulur (sabit aydınlatmada en pratik). ADAPTIVE: eşiği bölgesel hesaplar (aydınlatma değişkense). MANUAL: eşik değerini siz girersiniz.",
        ),
        ParamSpec(
            "threshold_value",
            ParamType.INT,
            default=127,
            min=0,
            max=255,
            label="Eşik Değeri (Manuel)",
            help="MANUAL modda kullanılır: bu değerin üzerindeki pikseller beyaz, altındakiler siyah olur.",
        ),
        ParamSpec(
            "adaptive_block_size",
            ParamType.INT,
            default=11,
            min=3,
            max=99,
            step=2,
            label="Adaptive Blok Boyutu",
            help="ADAPTIVE modda her pikselin eşiğinin hesaplandığı komşuluk boyutu (tek sayı olmalı).",
        ),
        ParamSpec(
            "adaptive_c",
            ParamType.INT,
            default=2,
            min=-50,
            max=50,
            label="Adaptive C",
            help="ADAPTIVE modda hesaplanan yerel eşikten çıkarılan sabit; eşiği ince ayarlamak için kullanılır.",
        ),
        ParamSpec(
            "invert",
            ParamType.BOOL,
            default=False,
            label="Tersini Al",
            help="Siyah/beyazı yer değiştirir (ör. koyu zemin üzerinde açık renkli nesne tespit ederken kullanışlıdır).",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

        if params.get("clahe_enabled", False):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)

        result = gray
        if params.get("threshold_enabled", False):
            mode = params.get("threshold_mode", "OTSU")
            if mode == "OTSU":
                _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif mode == "ADAPTIVE":
                block = _ensure_odd(max(3, params.get("adaptive_block_size", 11)))
                result = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY,
                    block,
                    int(params.get("adaptive_c", 2)),
                )
            else:
                _, result = cv2.threshold(
                    gray, int(params.get("threshold_value", 127)), 255, cv2.THRESH_BINARY
                )

        if params.get("invert", False):
            result = cv2.bitwise_not(result)

        return {"image": result}


@registry.register
class RgbOp:
    id = "color.rgb"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "white_balance_enabled",
            ParamType.BOOL,
            default=False,
            label="Otomatik Beyaz Dengesi",
            help="Gray-World algoritmasıyla sahnenin renk sıcaklığını nötrler; farklı endüstriyel lamba/LED renklerinden gelen sapmayı giderir.",
        ),
        ParamSpec(
            "r_gain",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            label="Kırmızı Kazancı",
            help="Kırmızı kanalı bu katsayıyla çarpar; renk dengesini elle ince ayarlamak için kullanılır.",
        ),
        ParamSpec(
            "g_gain",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            label="Yeşil Kazancı",
            help="Yeşil kanalı bu katsayıyla çarpar.",
        ),
        ParamSpec(
            "b_gain",
            ParamType.FLOAT,
            default=1.0,
            min=0.0,
            max=3.0,
            step=0.05,
            label="Mavi Kazancı",
            help="Mavi kanalı bu katsayıyla çarpar.",
        ),
        ParamSpec(
            "channel_view",
            ParamType.ENUM,
            default="NONE",
            choices=["NONE", "R", "G", "B"],
            label="Tek Kanal Göster",
            help="Sadece seçilen renk kanalını gri tonlamalı olarak gösterir; bir rengin baskın olduğu bölgeleri ayırt etmek için kullanışlıdır.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        working = inputs["image"]

        if params.get("white_balance_enabled", False):
            working = _gray_world_white_balance(working)

        channel_view = params.get("channel_view", "NONE")
        if channel_view in ("R", "G", "B"):
            b, g, r = cv2.split(working)
            chan = {"R": r, "G": g, "B": b}[channel_view]
            return {"image": chan}

        r_gain = float(params.get("r_gain", 1.0))
        g_gain = float(params.get("g_gain", 1.0))
        b_gain = float(params.get("b_gain", 1.0))
        if r_gain == 1.0 and g_gain == 1.0 and b_gain == 1.0:
            return {"image": working}

        f = working.astype(np.float32)
        b, g, r = cv2.split(f)
        b *= b_gain
        g *= g_gain
        r *= r_gain
        return {"image": np.clip(cv2.merge([b, g, r]), 0, 255).astype(np.uint8)}


@registry.register
class HsvOp:
    id = "color.hsv"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [PortSpec("image", PortType.IMAGE)]
    params = [
        ParamSpec(
            "shadow_removal",
            ParamType.BOOL,
            default=False,
            label="Gölge Giderme (V Kanalı)",
            help="Parlaklık (V) kanalını histogram eşitler; gölge/parlaklık farklarının renk tespitini bozmasını azaltır.",
        ),
        ParamSpec(
            "apply_mask",
            ParamType.BOOL,
            default=False,
            label="Renk Aralığı Maskesi",
            help="Sadece aşağıdaki H/S/V aralığındaki pikselleri ORİJİNAL renkleriyle bırakır, "
            "aralık dışındakileri siyah yapar; belirli bir renkteki nesneyi (ör. yeşil etiket) "
            "ayıklamak için kullanılır. Görüntü RENKLİ kalır (gri/siyah-beyaza dönüşmez) — "
            "sadece aralık dışı pikseller siyaha boyanır.",
        ),
        ParamSpec(
            "h_min", ParamType.INT, default=0, min=0, max=179, label="H Min", help="Renk tonu alt sınırı (0-179)."
        ),
        ParamSpec(
            "h_max", ParamType.INT, default=179, min=0, max=179, label="H Max", help="Renk tonu üst sınırı (0-179)."
        ),
        ParamSpec(
            "s_min", ParamType.INT, default=0, min=0, max=255, label="S Min", help="Doygunluk alt sınırı."
        ),
        ParamSpec(
            "s_max", ParamType.INT, default=255, min=0, max=255, label="S Max", help="Doygunluk üst sınırı."
        ),
        ParamSpec(
            "v_min", ParamType.INT, default=0, min=0, max=255, label="V Min", help="Parlaklık alt sınırı."
        ),
        ParamSpec(
            "v_max", ParamType.INT, default=255, min=0, max=255, label="V Max", help="Parlaklık üst sınırı."
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        image = inputs["image"]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        if params.get("shadow_removal", False):
            h, s, v = cv2.split(hsv)
            v = cv2.equalizeHist(v)
            hsv = cv2.merge([h, s, v])

        corrected_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        if params.get("apply_mask", False):
            lower = np.array(
                [params.get("h_min", 0), params.get("s_min", 0), params.get("v_min", 0)], dtype=np.uint8
            )
            upper = np.array(
                [params.get("h_max", 179), params.get("s_max", 255), params.get("v_max", 255)],
                dtype=np.uint8,
            )
            mask = cv2.inRange(hsv, lower, upper)
            # Gerçek kullanıcı raporu: "hsv filtresi otomatik siyah beyaza çeviriyor
            # çevirmemesi lazım" -- eskiden burada ham `mask` (tek kanallı 0/255 ikili
            # görüntü) döndürülüyordu, yani aralık maskesi AÇILINCA görüntü koşulsuz olarak
            # siyah/beyaza dönüyordu. `cv2.bitwise_and` ile SADECE aralık DIŞINDAKİ pikseller
            # siyaha boyanır, aralık İÇİNDEKİ pikseller (dolayısıyla görüntünün RENGİ) korunur.
            return {"image": cv2.bitwise_and(corrected_bgr, corrected_bgr, mask=mask)}

        return {"image": corrected_bgr}
