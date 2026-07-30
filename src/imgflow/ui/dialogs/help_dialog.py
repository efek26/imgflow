"""Kalibrasyon iş akışını (Kamera Ayarları -> Lens -> Yükseklik-Ölçek -> Ölçüm) anlatan,
uygulama içinden erişilebilir statik yardım penceresi — non-modal, harici state taşımaz.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

_HELP_HTML = """
<h2>Kalibrasyon İş Akışı Kılavuzu</h2>
<p><b>Lens Kalibrasyonu ile Yükseklik-Ölçek Kalibrasyonu farklı şeyleri ölçer, ikisi de
checkerboard kullanabilir ama BİRBİRİNİN YERİNE GEÇMEZ:</b></p>
<ul>
<li><b>Lens Kalibrasyonu</b> = kameranın kendi özelliği (odak uzaklığı + distorsiyon).
Kamera/lens/zoom değişmediği sürece geçerli kalır; tahtayı farklı açı/mesafelerde
göstererek BİRÇOK kare gerektirir.</li>
<li><b>Yükseklik-Ölçek Kalibrasyonu</b> = kurulumun özelliği ("bu kamera şu an banda göre
ne kadar uzakta/açılı duruyor" &rarr; 1 piksel kaç mm). Kamera hareket ettirilirse
(yükseklik/açı değişirse) HER SEFERİNDE tekrarlanması gerekir.</li>
</ul>
<p><b>Yeni bir kurulum için önerilen (hızlı) yol:</b> Kamerayı aç &rarr; Lens Kalibrasyonu
&mdash; TEK diyalogda hem intrinsics'i hem de ölçeği/düzlemi çözer, ayrı bir Yükseklik-Ölçek
adımına GEREK KALMAZ (bkz. adım 3, "Referans (Bant Seviyesi)"). Yükseklik-Ölçek Kalibrasyonu
diyaloğu ESKİ/manuel bir yedek yöntemdir &mdash; sadece checkerboard'ınız yoksa (2 noktaya
elle tıklama) ya da lens kalibrasyonu yapmadan kaba bir ölçek yeterliyse kullanın.</p>

<h3>1) Kamerayı Açma</h3>
<p><b>Kamera</b> menüsü &rarr; <i>USB Kamera Aç...</i> / <i>GigE/Basler Kamera Aç...</i> /
<i>Video Dosyası Aç...</i>. Sonraki tüm kalibrasyon diyalogları canlı kareyi bu kaynaktan
alır; kamera açık değilse diyaloglar açılmaz.</p>

<h3>2) Kamera Ayarları Sekmesi</h3>
<p><b>Kamera</b> menüsü &rarr; <i>Kamera Ayarları...</i> sağ sütunda ayrı bir sekmeye
geçer. Bu sekme <b>SADECE</b> gerçek bir Basler GigE/USB3 kamera bağlandığında etkinleşir
&mdash; USB webcam veya video dosyasında bu ayarlar (GenICam) mevcut olmadığı için menü
aksiyonu devre dışı kalır.</p>
<p>Kategoriler:</p>
<ul>
<li><b>Görüntü Kalitesi</b> &mdash; pozlama, kazanç, siyah seviyesi, beyaz dengesi.</li>
<li><b>Görüntü Formatı</b> &mdash; piksel formatı, çözünürlük/offset, binning/decimation,
aynalama.</li>
<li><b>Yakalama Kontrolü</b> &mdash; kare hızı ve tetikleme (Trigger) ayarları.</li>
<li><b>Dijital I/O</b> &mdash; kameranın fiziksel giriş/çıkış pinlerini (Line1, Line2, ...)
yapılandırır.</li>
<li><b>Aktarım Katmanı</b> &mdash; ağ/bant genişliği ayarları; çoğu alan sadece GigE
kameralarda görünür, USB3'te gizli kalabilir.</li>
<li><b>Kullanıcı Setleri</b> &mdash; kamera belleğinde saklı ayar profillerini kaydetme/
açılışta otomatik yükleme.</li>
</ul>
<p>Her alanın üzerine gelindiğinde (tooltip) o alanın ne işe yaradığını açıklayan bir
ipucu görünür.</p>

<h3>3) Lens Kalibrasyonu (intrinsics + ölçek/düzlem, önerilen tek durak)</h3>
<p><b>Kamera</b> menüsü &rarr; <i>Lens Kalibrasyonu...</i> &mdash; kamera açık olmalı.
Tahta türünü seç (Klasik Checkerboard veya ChArUco), tahtayı farklı açı/mesafelerden
göstererek <i>Kare Yakala</i> ile birkaç kare topla, <i>Kalibre Et</i> ile RMS hatayı gör.
Ardından galeride, checkerboard'ı ölçülecek ürünün durduğu düzleme (bandın üstüne) koyarak
çektiğin BİR kareyi <i>Referans (Bant Seviyesi)</i> olarak işaretle &mdash; bu, kamera
düzleme dik olmasa (Açılı montaj) bile KONUM-BAĞIMSIZ doğru <i>mm_per_px</i> üretir (düzlem
homografisi). İstersen isim vererek <i>Profili Kaydet</i>.</p>

<h3>4) Yükseklik-Ölçek Kalibrasyonu (ESKİ/manuel yedek yöntem)</h3>
<p><b>Araçlar</b> menüsü &rarr; <i>Yükseklik Kalibrasyonu (Öğretme, elle)...</i>. Lens
Kalibrasyonu'nda "Referans (Bant Seviyesi)" zaten işaretlendiyse bu adıma GEREK YOK. Sadece
checkerboard'sız kaba bir ölçek (2 noktaya tıklayıp gerçek mesafeyi elle girme) ya da lens
kalibrasyonu yapmadan Dik montajda basit bir piksel-aralığı ölçeği gerekiyorsa kullan. Kamera
Montajı (Dik/Açılı) seç &mdash; açılı montaj lens kalibrasyonu ister. <i>Kare Yakala</i>
sonrası ya kanvasta bilinen uzunluğa iki nokta tıklayıp yükseklik/gerçek mesafeyi elle gir
(<i>Nokta Ekle</i>), ya da tahtayı <i>Tahtayı Algıla</i> ile otomatik ölçtür. En az 2 farklı
yükseklikte nokta topladıktan sonra <i>Modeli Hesapla</i>, sonra istersen
<i>Profili Kaydet</i>.</p>

<h3>5) Kullanım</h3>
<p><b>Araçlar</b> &rarr; <i>Aktif Yükseklik Ayarla...</i> ile o anki yüksekliği gir,
<b>Araçlar</b> &rarr; <i>Ölçüm Aracı...</i> ile aktif mm/px ölçeğiyle görüntü üzerinde
gerçek dünya ölçümü yap.</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Kalibrasyon Kılavuzu")
        self.setModal(False)
        self.resize(560, 640)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(False)
        browser.setHtml(_HELP_HTML)

        layout = QVBoxLayout(self)
        layout.addWidget(browser)
