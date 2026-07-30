"""Bağlı bileşen (connected components) etiketleme operatörü."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.params import ParamSpec, ParamType
from imgflow.core.types import PortSpec, PortType
from imgflow.operators import registry


@registry.register
class ConnectedComponentsOp:
    id = "segment.connected_components"
    inputs = [PortSpec("image", PortType.IMAGE)]
    outputs = [
        PortSpec(
            "labels",
            PortType.IMAGE,
            description="Her pikselin bileşen etiketini taşıyan int32 harita (0 = arka plan).",
        ),
        PortSpec("count", PortType.SCALAR, description="Arka plan hariç bileşen sayısı."),
    ]
    params = [
        ParamSpec("connectivity", ParamType.ENUM, default="8", choices=["4", "8"], label="Komşuluk"),
        ParamSpec(
            "polarity",
            ParamType.ENUM,
            default="white",
            choices=["white", "black", "both"],
            label="Nesne Polaritesi",
            help="Hangi bölgelerin 'nesne' sayılacağı: 'white' = açık/beyaz bölgeler "
            "(varsayılan), 'black' = koyu/siyah bölgeler (ör. açık bir bant üzerindeki koyu "
            "ürünler), 'both' = ikisi de AYRI bileşenler olarak etiketlenir (hiçbir piksel "
            "arka plan kalmaz). Aksi halde koyu ürünleri ölçmek için önce Eşikleme'yi ters "
            "çevirmek gerekirdi.",
        ),
        ParamSpec(
            "max_area_ratio",
            ParamType.FLOAT,
            default=0.0,
            min=0.0,
            max=1.0,
            label="Maks. Alan Oranı",
            help="Bir bileşen, görüntü alanının bu orandan daha büyük bir kısmını kaplıyorsa "
            "arka plan (ör. yanlışlıkla eşiklenmiş bant/zemin) sayılıp elenir — otomatik "
            "ölçümün zemini bir 'ürün' olarak ölçmesine karşı güvenlik ağı. 0 = kapalı "
            "(varsayılan), hiçbir bileşen alanına göre elenmez.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        connectivity = int(params.get("connectivity", "8"))
        polarity = str(params.get("polarity", "white"))
        max_area_ratio = float(params.get("max_area_ratio", 0.0))
        image = inputs["image"]
        gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Girdi zaten ikili (değerleri {0,255}'in alt kümesi) değilse -- ör. Eşiklemesiz
        # Siyah-Beyaz, RGB veya HSV'nin maskesiz çıktısı doğrudan bağlanmışsa -- sıfırdan
        # farklı HER piksel (ör. koyu bir arka planın gri değeri)ön plan sayılıp neredeyse
        # tüm görüntü tek bir "dev bölge" olarak algılanır. Bunu önlemek için otomatik Otsu
        # eşiklemesi uygulanır; zaten {0,255} olan bir girdide (kullanıcı bilinçli olarak
        # Eşikleme eklediyse) davranış değişmez. Sadece "iki farklı değer var" yeterli
        # değildir -- o iki değer tam olarak 0 ve 255 olmalı, aksi halde (ör. {40,117} gibi)
        # her ikisi de "sıfırdan farklı" sayılıp yine tek dev bölge oluşurdu.
        unique_values = set(np.unique(gray).tolist())
        if unique_values <= {0, 255}:
            binary = gray
        else:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        if polarity == "black":
            binary = cv2.bitwise_not(binary)

        if polarity == "both":
            # Açık VE koyu bölgeler ayrı ayrı etiketlenir; koyu etiketler beyazlarınkiyle
            # ÇAKIŞMASIN diye kaydırılır, böylece her bölge benzersiz bir kimlik alır ve
            # görüntüde arka plan (0) kalan piksel olmaz.
            num_white, labels_white = cv2.connectedComponents(binary, connectivity=connectivity)
            num_black, labels_black = cv2.connectedComponents(
                cv2.bitwise_not(binary), connectivity=connectivity
            )
            shifted_black = np.where(labels_black > 0, labels_black + (num_white - 1), 0)
            labels = np.where(labels_white > 0, labels_white, shifted_black)
            num_labels = (num_white - 1) + (num_black - 1) + 1
        else:
            num_labels, labels = cv2.connectedComponents(binary, connectivity=connectivity)

        excluded = 0
        if max_area_ratio > 0:
            # Eskiden her etiket için `labels == label` ile TÜM görüntüyü tarayan bir Python
            # döngüsüydü (N etiket -> N tam-kare taraması) — canlı kamera akışında (bkz.
            # `analysis.region_props`'taki aynı sınıf performans düzeltmesi) gereksiz yavaştı.
            # `np.bincount` tek geçişte her etiketin piksel sayısını verir, eleme de tek
            # vektörel `np.where` ile yapılır.
            threshold_pixels = max_area_ratio * labels.size
            counts = np.bincount(labels.ravel(), minlength=num_labels)
            exclude_mask = counts > threshold_pixels
            exclude_mask[0] = False  # arka plan (0) hiçbir zaman "elenen bileşen" sayılmaz
            excluded = int(np.count_nonzero(exclude_mask))
            if excluded:
                labels = np.where(exclude_mask[labels], 0, labels)

        return {"labels": labels, "count": num_labels - 1 - excluded}
