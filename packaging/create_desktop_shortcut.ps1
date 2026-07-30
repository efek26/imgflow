# Masaüstüne "EK" adında, özel ikonlu bir kısayol (.lnk) oluşturur.
#
# BİLEREK run_imgflow.bat'a DEĞİL, doğrudan .venv\Scripts\imgflow.exe'ye işaret eder --
# .bat dosyaları Windows'ta her zaman önce bir cmd.exe konsol penceresi üzerinden çalışır
# (gerçek kullanıcı raporu: "uygulamayla beraber arkada bir konsol açılıyor"). imgflow.exe
# zaten bir `gui-scripts` stub'ı (pencere-sistemi, konsolsuz) olduğundan, Explorer'ın onu
# doğrudan bir .lnk üzerinden başlatması hiçbir konsol açmaz.
#
# Tekrar çalıştırmak güvenlidir -- var olan "EK.lnk" kısayolunu günceller, üzerine yazar.
#
# Kullanım: PowerShell'de bu dosyayı çalıştırın (repo kökünden ya da herhangi bir yerden):
#   powershell -ExecutionPolicy Bypass -File packaging\create_desktop_shortcut.ps1

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$TargetExe = Join-Path $RepoRoot ".venv\Scripts\imgflow.exe"
$IconPath = Join-Path $RepoRoot "src\imgflow\resources\icons\ek_icon.ico"
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "EK.lnk"

if (-not (Test-Path $TargetExe)) {
    throw "Bulunamadi: $TargetExe -- once 'pip install -e .' ile .venv kurulu oldugundan emin olun."
}
if (-not (Test-Path $IconPath)) {
    throw "Bulunamadi: $IconPath -- once 'python packaging\generate_icon.py' calistirin."
}

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $TargetExe
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.IconLocation = $IconPath
$Shortcut.Description = "EK - goruntu isleme pipeline araci"
$Shortcut.Save()

Write-Host "Olusturuldu: $ShortcutPath"
