# EK uygulamasının kullanıcı verilerini (şekil modelleri, kalibrasyon profilleri, aydınlatma
# referansları, özel filtreler, ONNX modelleri, kamera ayarları) TEK bir klasörde toplar ya da
# başka bir bilgisayarda geri yükler.
#
# Neden gerekli: bu veriler İKİ AYRI yerde durur --
#   1) %USERPROFILE%\.imgflow\   -> şekil modelleri, aydınlatma referansları, özel filtreler,
#                                   ONNX modelleri (her bilgisayarın KENDİ klasörü)
#   2) uygulamanın çalışma klasörü -> calibration\, camera_settings\ (EK.exe'nin YANINDA aranır)
# Bu betik ikisini birden ele alır, böylece elle klasör aramanız gerekmez.
#
# Kullanım:
#   # Bu bilgisayardaki verileri flash belleğe topla:
#   powershell -ExecutionPolicy Bypass -File packaging\veri_aktar.ps1 -Topla E:\EK-veri
#
#   # Karşı bilgisayarda geri yükle (EK.exe'nin bulunduğu klasörü -ExeKlasoru ile verin):
#   powershell -ExecutionPolicy Bypass -File packaging\veri_aktar.ps1 -Yukle E:\EK-veri -ExeKlasoru E:\EK
#
# Yakalanan kareler (captures) ve loglar BİLEREK dahil edilmez -- onlarca MB tutarlar ve
# karşı tarafta bir işe yaramazlar. İstersen -Yakalamalar ile dahil edebilirsin.

[CmdletBinding(DefaultParameterSetName = "Topla")]
param(
    [Parameter(Mandatory = $true, ParameterSetName = "Topla", Position = 0)]
    [string]$Topla,

    [Parameter(Mandatory = $true, ParameterSetName = "Yukle", Position = 0)]
    [string]$Yukle,

    # Geri yüklerken calibration\ ve camera_settings\ klasörlerinin YAZILACAĞI yer: EK.exe'nin
    # bulunduğu klasör. Verilmezse bu iki klasör atlanır (diğerleri yine de yüklenir).
    [Parameter(ParameterSetName = "Yukle")]
    [string]$ExeKlasoru,

    [switch]$Yakalamalar
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$HomeData = Join-Path $env:USERPROFILE ".imgflow"

# "ev" = %USERPROFILE%\.imgflow altında; "yaninda" = EK.exe ile aynı klasörde aranır.
$Klasorler = @(
    @{ Ad = "shape_models";    Tur = "ev";      Aciklama = "Şekil modelleri" },
    @{ Ad = "flatfield";       Tur = "ev";      Aciklama = "Aydınlatma referansları" },
    @{ Ad = "custom_filters";  Tur = "ev";      Aciklama = "Özel filtreler" },
    @{ Ad = "onnx_models";     Tur = "ev";      Aciklama = "ONNX modelleri" },
    @{ Ad = "calibration";     Tur = "yaninda"; Aciklama = "Kalibrasyon profilleri" },
    @{ Ad = "camera_settings"; Tur = "yaninda"; Aciklama = "Kamera ayarları" }
)
if ($Yakalamalar) {
    $Klasorler += @{ Ad = "captures"; Tur = "ev"; Aciklama = "Yakalanan kareler" }
}

function Get-KaynakYol($k) {
    if ($k.Tur -eq "ev") { return Join-Path $HomeData $k.Ad }
    return Join-Path $RepoRoot $k.Ad
}

function Copy-Klasor($Kaynak, $Hedef, $Aciklama) {
    if (-not (Test-Path $Kaynak)) {
        Write-Host ("  {0,-28} yok, atlandi" -f $Aciklama) -ForegroundColor DarkGray
        return
    }
    $dosyalar = @(Get-ChildItem -Path $Kaynak -File -Recurse -ErrorAction SilentlyContinue)
    if ($dosyalar.Count -eq 0) {
        Write-Host ("  {0,-28} bos, atlandi" -f $Aciklama) -ForegroundColor DarkGray
        return
    }
    New-Item -ItemType Directory -Force -Path $Hedef | Out-Null
    Copy-Item -Path (Join-Path $Kaynak "*") -Destination $Hedef -Recurse -Force
    $mb = [math]::Round((($dosyalar | Measure-Object -Property Length -Sum).Sum / 1MB), 1)
    Write-Host ("  {0,-28} {1} dosya ({2} MB)" -f $Aciklama, $dosyalar.Count, $mb) -ForegroundColor Green
}

if ($PSCmdlet.ParameterSetName -eq "Topla") {
    Write-Host "EK verileri toplaniyor -> $Topla`n" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path $Topla | Out-Null
    foreach ($k in $Klasorler) {
        Copy-Klasor (Get-KaynakYol $k) (Join-Path $Topla $k.Ad) $k.Aciklama
    }
    Write-Host "`nTamam. Bu klasoru flash bellege kopyalayin." -ForegroundColor Cyan
    Write-Host "Karsi bilgisayarda geri yuklemek icin:" -ForegroundColor Cyan
    Write-Host "  powershell -ExecutionPolicy Bypass -File veri_aktar.ps1 -Yukle <bu klasor> -ExeKlasoru <EK.exe klasoru>"
}
else {
    if (-not (Test-Path $Yukle)) { throw "Bulunamadi: $Yukle" }
    Write-Host "EK verileri geri yukleniyor <- $Yukle`n" -ForegroundColor Cyan
    foreach ($k in $Klasorler) {
        $kaynak = Join-Path $Yukle $k.Ad
        if ($k.Tur -eq "ev") {
            $hedef = Join-Path $HomeData $k.Ad
        }
        elseif ($ExeKlasoru) {
            $hedef = Join-Path $ExeKlasoru $k.Ad
        }
        else {
            Write-Host ("  {0,-28} atlandi (-ExeKlasoru verilmedi)" -f $k.Aciklama) -ForegroundColor Yellow
            continue
        }
        Copy-Klasor $kaynak $hedef $k.Aciklama
    }
    Write-Host "`nTamam. EK.exe'yi acabilirsiniz." -ForegroundColor Cyan
}
