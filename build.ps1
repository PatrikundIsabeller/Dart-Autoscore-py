param(
    [string]$Version = "0.1.0",
    [string]$IconPng = "icon.png",
    [string]$Spec = ".\Triple One Controller.spec"
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

Write-Host "==> Aktivere venv (falls nötig)..." -ForegroundColor Cyan
if (-not (Test-Path ".\.venv\Scripts\Activate.ps1")) {
    Write-Host "WARN: venv fehlt. Bitte vorher erstellen/aktivieren." -ForegroundColor Yellow
}
else {
    . .\.venv\Scripts\Activate.ps1
}

# 1) Optional: PNG -> ICO
if (Test-Path $IconPng) {
    Write-Host "==> Konvertiere $IconPng -> icon.ico" -ForegroundColor Cyan
    pip show pillow | Out-Null 2>$null
    if ($LASTEXITCODE -ne 0) { pip install pillow }
    @"
from PIL import Image, ImageOps
src = Image.open("$IconPng").convert("RGBA")
if src.width != src.height:
    side = max(src.width, src.height)
    canvas = Image.new("RGBA", (side, side), (0,0,0,0))
    canvas.paste(src, ((side - src.width)//2, (side - src.height)//2))
    src = canvas
src = ImageOps.contain(src, (256,256))
src.save("icon.ico", sizes=[(256,256),(128,128),(64,64),(48,48),(32,32),(16,16)])
print("icon.ico erzeugt.")
"@ | Set-Content -Path .\make_icon.py -Encoding UTF8
    python .\make_icon.py
}

# 2) version.txt erstellen/aktualisieren (optional)
$versionFile = ".\version.txt"
if (-not (Test-Path $versionFile)) {
    @"
VSVersionInfo(
  ffi=FixedFileInfo(filevers=(${Version.Replace('.', ',')},0), prodvers=(${Version.Replace('.', ',')},0),
                    mask=0x3f, flags=0x0, OS=0x40004, fileType=0x1, subtype=0x0, date=(0,0)),
  kids=[
    StringFileInfo([StringTable('040704B0', [
      StringStruct('CompanyName', 'Triple One'),
      StringStruct('FileDescription', 'Triple One Controller'),
      StringStruct('FileVersion', '$Version'),
      StringStruct('OriginalFilename', 'Triple One Controller.exe'),
      StringStruct('ProductName', 'Triple One Controller'),
      StringStruct('ProductVersion', '$Version')
    ])]),
    VarFileInfo([VarStruct('Translation', [0x0407, 1200])])
  ]
)
"@ | Set-Content -Path $versionFile -Encoding UTF8
}
else {
    # Versionsnummern in bestehender version.txt ersetzen
    (Get-Content $versionFile) `
        -replace "filevers=\([0-9, ]+\)", ("filevers=(" + $Version.Replace('.', ',') + ",0)") `
        -replace "prodvers=\([0-9, ]+\)", ("prodvers=(" + $Version.Replace('.', ',') + ",0)") `
        -replace "StringStruct\('FileVersion', '([^']+)'\)", ("StringStruct('FileVersion', '$Version')") `
        -replace "StringStruct\('ProductVersion', '([^']+)'\)", ("StringStruct('ProductVersion', '$Version')") `
  | Set-Content $versionFile -Encoding UTF8
}

# 3) Clean build/dist
Write-Host "==> Clean build/dist" -ForegroundColor Cyan
Remove-Item -Recurse -Force .\build, .\dist -ErrorAction SilentlyContinue

# 4) PyInstaller via .spec
Write-Host "==> Baue mit Spec: $Spec" -ForegroundColor Cyan
pyinstaller $Spec

# 5) Release kopieren (robust: Quelle -> Ziel spiegeln)
$src = ".\dist\Triple One Controller"
$release = ".\release\Triple One Controller_v$Version"
New-Item -Force -ItemType Directory $release | Out-Null
robocopy $src $release /MIR /NFL /NDL /NJH /NJS /NP | Out-Null
Write-Host "==> Fertig! Release liegt in: $release" -ForegroundColor Green
