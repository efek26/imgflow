"""Kalibrasyon iş akışını (Kamera Ayarları -> Lens -> Yükseklik-Ölçek -> Ölçüm) anlatan,
uygulama içinden erişilebilir statik yardım penceresi — non-modal, harici state taşımaz.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QTextBrowser, QVBoxLayout

_HELP_HTML = """
<h2>Kalibrasyon İş Akışı Kılavuzu</h2>
<p><b>Genel sıra:</b> Kamerayı aç &rarr; (açılı montajda zorunlu) Lens Kalibrasyonu &rarr;
Yükseklik-Ölçek Kalibrasyonu &rarr; Aktif Yükseklik Ayarla &rarr; Ölçüm Aracı.</p>

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

<h3>3) Lens Kalibrasyonu (intrinsics)</h3>
<p><b>Kamera</b> menüsü &rarr; <i>Lens Kalibrasyonu...</i> &mdash; kamera açık olmalı.
Tahta türünü seç (Klasik Checkerboard veya ChArUco), tahtayı farklı açı/mesafelerden
göstererek <i>Kare Yakala</i> ile birkaç kare topla, <i>Kalibre Et</i> ile RMS hatayı gör,
istersen isim vererek <i>Profili Kaydet</i>.</p>

<h3>4) Yükseklik-Ölçek Kalibrasyonu</h3>
<p><b>Araçlar</b> menüsü &rarr; <i>Yükseklik Kalibrasyonu (Öğretme)...</i>. Kamera Montajı
(Dik/Açılı) seç &mdash; açılı montaj lens kalibrasyonu ister. <i>Kare Yakala</i> sonrası ya
kanvasta bilinen uzunluğa iki nokta tıklayıp yükseklik/gerçek mesafeyi elle gir
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
