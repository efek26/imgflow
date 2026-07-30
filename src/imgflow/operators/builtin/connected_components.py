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
            label="Ölçülecek Bölge",
            help="'Beyaz' (varsayılan, eski davranış): sadece eşiklemeden sonra parlak kalan "
            "bölgeler etiketlenir. 'Siyah': sadece koyu kalan bölgeler etiketlenir. 'Her İkisi': "
            "hem parlak hem koyu bölgeler AYRI etiketlerle (koyu bölgeler parlak "
            "bölgelerinkinden sonra gelen numaralarla) tek haritada birlikte döner. Siyah-Beyaz "
            "filtrede tersini alınca gölge/aydınlatma kaynaklı bozukluklarda hangi tarafın "
            "gerçek ürün olduğu değişebiliyorsa bu seçenekle yeniden Eşikleme/Ters Çevirme "
            "eklemeden düzeltilebilir.",
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
        max_area_ratio = float(params.get("max_area_ratio", 0.0))
        polarity = params.get("polarity", "white")
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

        if polarity == "both":
            # İki bağımsız `connectedComponents` çağrısı (biri parlak taraf, biri koyu taraf
            # `bitwise_not` ile) sonra TEK bir haritada birleştirilir. `labels_black` sadece
            # `binary`'nin 0 olduğu (yani `labels_white`'ın da 0/arka plan olduğu) piksellerde
            # sıfırdan farklı olduğundan iki harita KESİŞMEZ -- basit toplama yeterli, hücre
            # bazlı öncelik/üzerine yazma mantığı gerekmez.
            num_white, labels_white = cv2.connectedComponents(binary, connectivity=connectivity)
            num_black, labels_black = cv2.connectedComponents(
                cv2.bitwise_not(binary), connectivity=connectivity
            )
            offset = num_white - 1
            labels = labels_white + np.where(labels_black > 0, labels_black + offset, 0)
            num_labels = num_white + num_black - 1
        else:
            source = cv2.bitwise_not(binary) if polarity == "black" else binary
            num_labels, labels = cv2.connectedComponents(source, connectivity=connectivity)

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
