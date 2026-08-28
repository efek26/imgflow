# imgflow

**imgflow**, endüstriyel kalite kontrol ve otomasyon süreçleri için geliştirilmiş, modüler bir bilgisayarlı görü (computer vision) ardışık düzeni ve masaüstü arayüzüdür. Kullanıcıların görüntü işleme operatörlerini zincirlemesine, parametreleri gerçek zamanlı ayarlamasına ve kalite kontrol testlerini otomatik hale getirmesine olanak tanır.

---

## 📸 Ekran Görüntüleri

<img width="1535" height="864" alt="imgflow Arayüzü" src="https://github.com/user-attachments/assets/f2894c13-d9df-479c-8657-e9d1f2a858aa" />

<img width="1531" height="853" alt="imgflow Pipeline Akışı" src="https://github.com/user-attachments/assets/10e2f6a7-c9c5-4af5-8cf3-56b3fd1e79d9" />

---

## 🚀 Temel Özellikler

* **Modüler Görüntü İşleme Pipeline'ı:** ROI seçimi, eşikleme (thresholding), filtreleme ve morfolojik operatörler.
* **Gelişmiş Analiz & Tespit:** Doku analizi (GLCM), bağlı bileşenler (connected components), şekil eşleştirme ve YOLO / ONNX model entegrasyonu.
* **Modern GUI:** PySide6 tabanlı sezgisel ve hızlı kontrol paneli.
* **Reçete & Konfigürasyon Yönetimi:** JSON formatında işlem adımlarını kaydetme ve farklı üretim hatları için profiller yükleme.
* **Kamera ve Kalibrasyon Desteği:** Kamera parametre yönetimi ve kalibrasyon araçları.

---

## 🛠️ Kurulum & Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone [https://github.com/efek26/imgflow.git](https://github.com/efek26/imgflow.git)
cd imgflow
