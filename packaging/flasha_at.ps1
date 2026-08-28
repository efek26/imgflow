# EK uygulamasını + kullanıcı verilerini + karşı taraf için yönergeleri TEK KOMUTLA bir flash
# belleğe hazırlar.
#
# Kullanım (repo kökünden, flash bellek takılıyken):
#   powershell -ExecutionPolicy Bypass -File packaging\flasha_at.ps1 -Surucu E:
#
# Flash bellekte oluşan yapı:
#   E:\EK\             -> uygulama (EK.exe'ye çift tıklanır)
#   E:\EK-veri\        -> şekil modelleri, kalibrasyon profilleri, aydınlatma referansları...
#   E:\veri_aktar.ps1  -> karşı bilgisayarda veriyi geri yükleyen betik
#   E:\OKUBENI.txt     -> karşı taraf için adım adım yönerge
#
# `veri_aktar.ps1`'in flash belleğe KOPYALANMASI şart: karşı bilgisayarda bu repo olmayacağı
# için, veriyi geri yükleyecek betiğin uygulamayla birlikte gitmesi gerekir.

param(
    [Parameter(Mandatory = $true)]
    [string]$Surucu,

    # Yakalanan kareleri de (onlarca MB) dahil et.
    [switch]$Yakalamalar
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Kaynak = Join-Path $RepoRoot "dist\EK"

if (-not (Test-Path (Join-Path $Kaynak "EK.exe"))) {
    throw "Bulunamadi: $Kaynak\EK.exe -- once paketi derleyin: pyinstaller packaging\EK.spec --noconfirm"
}

$Surucu = $Surucu.TrimEnd('\')
if (-not (Test-Path "$Surucu\")) {
    throw "Surucu bulunamadi: $Surucu -- flash bellek takili mi? Harfi 'Bu Bilgisayar'dan kontrol edin."
}

$HedefApp = Join-Path $Surucu "EK"
$HedefVeri = Join-Path $Surucu "EK-veri"

# Yer kontrolu: kopyalama yarida kesilirse calismayan bir paket kalir, bu yuzden ONCEDEN bak.
$gerekliMB = [math]::Round(((Get-ChildItem $Kaynak -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 0)
$bosMB = [math]::Round(((Get-CimInstance Win32_LogicalDisk -Filter "DeviceID='$Surucu'").FreeSpace / 1MB), 0)
Write-Host "Uygulama boyutu: $gerekliMB MB   |   $Surucu bos alan: $bosMB MB" -ForegroundColor Cyan
if ($bosMB -lt ($gerekliMB + 200)) {
    throw "Yetersiz alan: $Surucu uzerinde en az $($gerekliMB + 200) MB bos yer gerekiyor."
}

Write-Host "`n[1/5] Uygulama kopyalaniyor -> $HedefApp" -ForegroundColor Cyan
if (Test-Path $HedefApp) { Remove-Item $HedefApp -Recurse -Force }
Copy-Item -Path $Kaynak -Destination $HedefApp -Recurse -Force
Write-Host "  tamam" -ForegroundColor Green

Write-Host "`n[2/5] Kullanici verileri toplaniyor -> $HedefVeri" -ForegroundColor Cyan
$veriBetik = Join-Path $PSScriptRoot "veri_aktar.ps1"
if ($Yakalamalar) { & $veriBetik -Topla $HedefVeri -Yakalamalar } else { & $veriBetik -Topla $HedefVeri }

# Kalibrasyon profilleri ve kamera ayarlari EK.exe'nin YANINDA aranir (CWD'ye goreli, bkz.
# io_utils/calibration_store.py). Bunlari dogrudan oraya da koyuyoruz ki karsi taraf HICBIR
# betik calistirmadan, sadece EK.exe'ye cift tiklayarak kalibrasyonlu baslasin. Geri kalan
# veri (sekil modelleri, aydinlatma referanslari) %USERPROFILE%\.imgflow altina gitmek
# zorunda oldugu icin onlar icin veri_aktar.ps1 adimi hala gerekli.
Write-Host "`n[3/5] Kalibrasyon EK.exe'nin yanina yerlestiriliyor" -ForegroundColor Cyan
foreach ($ad in @("calibration", "camera_settings")) {
    $k = Join-Path $HedefVeri $ad
    if (Test-Path $k) {
        Copy-Item -Path $k -Destination (Join-Path $HedefApp $ad) -Recurse -Force
        Write-Host "  $ad kopyalandi" -ForegroundColor Green
    }
}

Write-Host "`n[4/5] Geri yukleme betigi kopyalaniyor -> $Surucu\veri_aktar.ps1" -ForegroundColor Cyan
Copy-Item -Path $veriBetik -Destination (Join-Path $Surucu "veri_aktar.ps1") -Force
Write-Host "  tamam" -ForegroundColor Green

Write-Host "`n[5/5] OKUBENI.txt yaziliyor" -ForegroundColor Cyan
# Metin, duzenlenebilir olmasi icin ayri bir sablon dosyada tutulur (bu betigin icine gomulu
# DEGIL) -- karsi tarafa gidecek yonergeyi degistirmek icin PowerShell'e dokunmak gerekmesin.
# {SURUCU} yer tutucusu, gercek flash bellek harfiyle degistirilir.
$sablon = Join-Path $PSScriptRoot "OKUBENI.txt"
if (-not (Test-Path $sablon)) { throw "Bulunamadi: $sablon" }
$okubeni = ([System.IO.File]::ReadAllText($sablon)).Replace("{SURUCU}", $Surucu)
[System.IO.File]::WriteAllText((Join-Path $Surucu "OKUBENI.txt"), $okubeni, [System.Text.UTF8Encoding]::new($true))
Write-Host "  tamam" -ForegroundColor Green

Write-Host "`nHAZIR. $Surucu icerigi:" -ForegroundColor Cyan
Get-ChildItem $Surucu | Select-Object Name, @{n='Tur';e={if ($_.PSIsContainer) {'klasor'} else {'dosya'}}} | Format-Table -AutoSize
Write-Host "Flash bellegi 'Donanimi guvenle kaldir' ile cikarin." -ForegroundColor Cyan
