# build.ps1 — Production build script for Gestura.exe
#
# What this does:
#   1. Validates the virtual environment and required model files exist
#   2. Installs PyInstaller into the venv if missing
#   3. Cleans previous build artifacts
#   4. Runs PyInstaller with gestura.spec (one-folder output)
#   5. Reports the output location and size
#
# Usage:
#   cd build
#   .\build.ps1
#
# Output: build\dist\Gestura\Gestura.exe
#
# Prerequisites:
#   - .venv\ must exist at the project root (run: python -m venv .venv)
#   - dependencies must be installed (run: pip install -r requirements.txt)
#   - model weights must exist:
#       training\Results\Run_3\gesture_model_300_inference.pt
#       training\Results\Fingerspelling\fingerspelling_model.pt

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir
$VenvPython  = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Gestura v2 — PyInstaller Build" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Validate virtual environment ──────────────────────────────────────
Write-Host "[1/5] Checking virtual environment..." -ForegroundColor Green
if (-not (Test-Path $VenvPython)) {
    Write-Host ""
    Write-Error "Virtual environment not found at: $VenvPython\n\nPlease create it first:\n    python -m venv .venv\n    .\\.venv\\Scripts\\Activate.ps1\n    pip install -r requirements.txt"
    exit 1
}
Write-Host "      OK: $VenvPython"

# ── Step 2: Validate required model files ─────────────────────────────────────
Write-Host "[2/5] Checking required model files..." -ForegroundColor Green

$GestureModel = Join-Path $ProjectRoot "training\Results\Run_3\gesture_model_300_inference.pt"
$FingerModel  = Join-Path $ProjectRoot "training\Results\Fingerspelling\fingerspelling_model.pt"
$LabelMap     = Join-Path $ProjectRoot "landmarks_300\label_map.json"

$MissingFiles = @()
if (-not (Test-Path $GestureModel)) { $MissingFiles += $GestureModel }
if (-not (Test-Path $FingerModel))  { $MissingFiles += $FingerModel }
if (-not (Test-Path $LabelMap))     { $MissingFiles += $LabelMap }

if ($MissingFiles.Count -gt 0) {
    Write-Host ""
    Write-Host "ERROR: Required files not found:" -ForegroundColor Red
    foreach ($f in $MissingFiles) {
        Write-Host "  MISSING: $f" -ForegroundColor Red
    }
    Write-Host ""
    Write-Host "Build cannot continue without these files." -ForegroundColor Red
    exit 1
}

Write-Host "      OK: gesture_model_300_inference.pt"
Write-Host "      OK: fingerspelling_model.pt"
Write-Host "      OK: label_map.json"

# ── Step 3: Install PyInstaller ───────────────────────────────────────────────
Write-Host "[3/5] Installing / verifying PyInstaller..." -ForegroundColor Green
& $VenvPython -m pip install pyinstaller --quiet
if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed to install PyInstaller."
    exit 1
}
$PyiVersion = & $VenvPython -m PyInstaller --version 2>&1
Write-Host "      OK: PyInstaller $PyiVersion"

# ── Step 4: Clean previous build ─────────────────────────────────────────────
Write-Host "[4/5] Cleaning previous build artifacts..." -ForegroundColor Green
$DistDir   = Join-Path $ScriptDir "dist"
$BuildTemp = Join-Path $ScriptDir "build_temp"
if (Test-Path $DistDir)   { Remove-Item $DistDir   -Recurse -Force; Write-Host "      Removed dist/" }
if (Test-Path $BuildTemp) { Remove-Item $BuildTemp -Recurse -Force; Write-Host "      Removed build_temp/" }

# ── Step 5: Run PyInstaller ────────────────────────────────────────────────────
Write-Host "[5/5] Running PyInstaller (this may take 3-8 minutes)..." -ForegroundColor Green
Write-Host ""
Set-Location $ScriptDir

& $VenvPython -m PyInstaller `
    --clean `
    --distpath "$DistDir" `
    --workpath "$BuildTemp" `
    "gestura.spec"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Error "PyInstaller failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# ── Report ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "==========================================================" -ForegroundColor Cyan
$ExePath    = Join-Path $DistDir "Gestura\Gestura.exe"
$FolderPath = Join-Path $DistDir "Gestura"

if (Test-Path $ExePath) {
    $ExeSizeMB    = [math]::Round((Get-Item $ExePath).Length / 1MB, 1)
    $FolderSizeMB = [math]::Round((Get-ChildItem $FolderPath -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 0)

    Write-Host "  BUILD SUCCESSFUL" -ForegroundColor Green
    Write-Host ""
    Write-Host "  Exe:    $ExePath" -ForegroundColor Cyan
    Write-Host "  Exe size:    ${ExeSizeMB} MB" -ForegroundColor Cyan
    Write-Host "  Folder size: ${FolderSizeMB} MB total" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "  To distribute: zip the entire dist\Gestura\ folder." -ForegroundColor Yellow
    Write-Host "  Users run Gestura.exe from inside that folder." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "  NOTE: pyvirtualcam (virtual camera) requires OBS Studio" -ForegroundColor Yellow
    Write-Host "        to be installed on the target machine separately." -ForegroundColor Yellow
} else {
    Write-Host "  WARNING: Gestura.exe not found in dist\Gestura\" -ForegroundColor Red
    Write-Host "  Check the PyInstaller output above for errors." -ForegroundColor Red
}
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host ""
