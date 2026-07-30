"""Bağlı bileşen (connected components) etiketleme operatörü."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from imgflow.core.auto_objects import _apply_auto_polarity
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
            choices=["white", "black", "both", "auto"],
            label="Ölçülecek Bölge",
            help="'Beyaz' (varsayılan, eski davranış): sadece eşiklemeden sonra parlak kalan "
            "bölgeler etiketlenir. 'Siyah': sadece koyu kalan bölgeler etiketlenir. 'Her İkisi': "
            "hem parlak hem koyu bölgeler AYRI etiketlerle (koyu bölgeler parlak "
            "bölgelerinkinden sonra gelen numaralarla) tek haritada birlikte döner. "
            "**'Otomatik'**: hangi tarafın ürün olduğunu görüntünün kendisinden çıkarır — "
            "görüntünün dış çerçevesine DEĞEN bölgeler arka plan sayılır (bkz. "
            "`core/auto_objects.py::_apply_auto_polarity`). **Ölçüm 'tüm şekli/zemini tek "
            "bölge olarak alıyorsa' seçilmesi gereken seçenek budur**: açık bir zemin "
            "üzerindeki KOYU ürünlerde 'Beyaz' polarite zeminin tamamını tek bir dev 'ürün' "
            "olarak etiketler. Gölge/eşit olmayan aydınlatma altında da dayanıklıdır "
            "(gerçek yakalamalarda ölçüldü), ama zaten elle bir Eşikleme adımıyla polariteyi "
            "kendiniz belirlediyseniz varsayılan 'Beyaz' daha öngörülebilirdir.",
        ),
        ParamSpec(
            "min_area_ratio",
            ParamType.FLOAT,
            default=0.0005,
            min=0.0,
            max=1.0,
            label="Min. Alan Oranı",
            help="Bir bileşen, görüntü alanının bu orandan daha KÜÇÜK bir kısmını kaplıyorsa "
            "gürültü sayılıp elenir. Gerçek kullanıcı raporu: \"direkt alan bulmaya çalışınca "
            "çok fazla bölge buluyor, o yüzden sistemi zora sokuyor\" — eşiklemeden kalan tek/"
            "birkaç piksellik binlerce leke etiketlenip `analysis.region_props` hepsini tek tek "
            "dolaşıyordu. Varsayılan %0.05 (bu yüzden 0'dan DEĞİL, çalışan bir değerden başlar) "
            "gerçek ürünleri elemeyecek kadar küçük, gürültüyü kesecek kadar büyüktür. "
            "`analysis.region_props`'un 'Min. Alan' alanının AKSİNE burada ORAN kullanılır: "
            "kalibrasyondan ve çözünürlükten bağımsızdır, kamera/görüntü değişince elle "
            "yeniden ayarlamak gerekmez. 0 = kapalı.",
        ),
        ParamSpec(
            "max_area_ratio",
            ParamType.FLOAT,
            default=0.9,
            min=0.0,
            max=1.0,
            label="Maks. Alan Oranı",
            help="Bir bileşen, görüntü alanının bu orandan daha büyük bir kısmını kaplıyorsa "
            "arka plan (ör. yanlışlıkla eşiklenmiş bant/zemin) sayılıp elenir — otomatik "
            "ölçümün zemini bir 'ürün' olarak ölçmesine karşı güvenlik ağı. Varsayılan 0.9: "
            "görüntünün %90'ından fazlasını kaplayan bir bileşen pratikte HER ZAMAN zemindir "
            "(gerçek kullanıcı raporu: \"bazen tüm şekli alıyor\"). NOT: bu bir güvenlik ağıdır, "
            "asıl çözüm değil — zemin ölçülüyorsa 'Ölçülecek Bölge'yi 'Otomatik' yapmak "
            "ürünlerin KENDİSİNİ etiketler, bu eleme ise sadece yanlış bölgeyi atar. "
            "'Her İkisi' polaritesiyle birlikte DİKKAT: tamamlayıcı (koyu) taraf tek parça ve "
            "görüntünün %90'ından büyükse o da elenir — iki tarafın ikisini de ölçmek "
            "istiyorsanız bu sınırı 0 yapın. 0 = kapalı.",
        ),
    ]

    def run(self, inputs: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
        connectivity = int(params.get("connectivity", "8"))
        # Fallback'ler ParamSpec varsayılanlarıyla AYNI tutulur: arayüz üzerinden kurulan
        # düğümlerde anahtar zaten dolu olur, bu değerler sadece programatik/eski (anahtarı
        # hiç taşımayan) çağrılarda devreye girer.
        min_area_ratio = float(params.get("min_area_ratio", 0.0005))
        max_area_ratio = float(params.get("max_area_ratio", 0.9))
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

        if polarity == "auto":
            # `shape_matching_dialog`'un eğitim-zamanı kontur tespitiyle AYNI ölçüt
            # (çerçeveye değen bileşen alanı) — tek bir yerde tutulur, iki farklı "hangi taraf
            # nesne" sezgiseli oluşmasın.
            binary = _apply_auto_polarity(binary)

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
        if min_area_ratio > 0 or max_area_ratio > 0:
            # Eskiden her etiket için `labels == label` ile TÜM görüntüyü tarayan bir Python
            # döngüsüydü (N etiket -> N tam-kare taraması) — canlı kamera akışında (bkz.
            # `analysis.region_props`'taki aynı sınıf performans düzeltmesi) gereksiz yavaştı.
            # `np.bincount` tek geçişte her etiketin piksel sayısını verir, eleme de tek
            # vektörel `np.where` ile yapılır.
            counts = np.bincount(labels.ravel(), minlength=num_labels)
            exclude_mask = np.zeros(counts.shape, dtype=bool)
            if max_area_ratio > 0:
                exclude_mask |= counts > max_area_ratio * labels.size
            if min_area_ratio > 0:
                # Min eleme, `region_props`'a hiç ULAŞMADAN gürültüyü keser — asıl performans
                # kazancı burada: aşağı akıştaki operatör binlerce yerine sadece gerçek
                # bileşenleri dolaşır.
                exclude_mask |= counts < min_area_ratio * labels.size
            exclude_mask[0] = False  # arka plan (0) hiçbir zaman "elenen bileşen" sayılmaz
            excluded = int(np.count_nonzero(exclude_mask))
            if excluded:
                labels = np.where(exclude_mask[labels], 0, labels)

        return {"labels": labels, "count": num_labels - 1 - excluded}
