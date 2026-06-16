# ============================================================
#  Run 3 Launch Script — GestureTransformer Anti-Overfitting
#  Usage: .\run3_launch.ps1
#  Or with custom args: .\run3_launch.ps1 --epochs 200
# ============================================================

param(
    [string]$LandmarksDir = "",
    [int]$Epochs = 500,
    [int]$BatchSize = 32,
    [int]$NumWorkers = 4,
    [switch]$QuickTest,
    [switch]$NoMixup,
    [switch]$AllowCPU,
    [string]$Resume = ""
)

$ScriptDir  = Split-Path -Parent $MyInvocation.MyCommand.Definition
$ModelDir   = Join-Path $ScriptDir "Model"
$VenvPython = Join-Path (Split-Path -Parent $ScriptDir) ".venv\Scripts\python.exe"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Gestura Run 3 — Anti-Overfitting Training Launch" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Verify Python venv ──────────────────────────────────────────────────
if (-not (Test-Path $VenvPython)) {
    Write-Host "  [ERROR] venv not found at: $VenvPython" -ForegroundColor Red
    Write-Host "  Run: python -m venv .venv  in the Gestura v2 root" -ForegroundColor Yellow
    exit 1
}
Write-Host "  [OK] Python venv found" -ForegroundColor Green

# ── 2. Check PyTorch + CUDA ────────────────────────────────────────────────
Write-Host "  Checking PyTorch + CUDA..." -ForegroundColor Yellow
$torchCheck = & $VenvPython -c @"
import torch, sys
v = torch.__version__
cuda = torch.cuda.is_available()
gpu = torch.cuda.get_device_name(0) if cuda else 'None'
print(f'TORCH={v}|CUDA={cuda}|GPU={gpu}')
"@

if ($torchCheck -match "TORCH=(.+)\|CUDA=(.+)\|GPU=(.+)") {
    $torchVer = $Matches[1]
    $cudaOK   = $Matches[2] -eq "True"
    $gpuName  = $Matches[3]

    if ($cudaOK) {
        Write-Host "  [OK] PyTorch $torchVer | CUDA | GPU: $gpuName" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] PyTorch $torchVer — CPU only (no CUDA)" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  *** GPU training is 10-20x faster. To enable CUDA: ***" -ForegroundColor Magenta
        Write-Host "  $VenvPython -m pip uninstall torch torchvision torchaudio -y" -ForegroundColor White
        Write-Host "  $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor White
        Write-Host ""
        if (-not $AllowCPU) {
            Write-Host "  Aborting. Use -AllowCPU to force CPU training (very slow)." -ForegroundColor Red
            exit 1
        }
        Write-Host "  -AllowCPU flag set — continuing on CPU (training will be slow)" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [ERROR] Could not query PyTorch. Is it installed in the venv?" -ForegroundColor Red
    Write-Host "  Run: $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124" -ForegroundColor Yellow
    exit 1
}

# ── 3. Locate landmarks_300 directory ─────────────────────────────────────
$GesturaV2Root = Split-Path -Parent $ScriptDir

$LandmarksCandidates = @(
    $LandmarksDir,
    (Join-Path $GesturaV2Root "landmarks_300"),
    "C:\landmarks_300",
    "D:\landmarks_300"
) | Where-Object { $_ -ne "" }

$FoundLandmarks = ""
foreach ($candidate in $LandmarksCandidates) {
    $manifest = Join-Path $candidate "manifest.json"
    if (Test-Path $manifest) {
        $FoundLandmarks = $candidate
        break
    }
}

if ($FoundLandmarks -eq "") {
    Write-Host ""
    Write-Host "  [ERROR] landmarks_300/ directory not found!" -ForegroundColor Red
    Write-Host ""
    Write-Host "  Searched:" -ForegroundColor Yellow
    foreach ($c in $LandmarksCandidates) {
        Write-Host "    - $c" -ForegroundColor Yellow
    }
    Write-Host ""
    Write-Host "  You need to extract landmarks first:" -ForegroundColor Yellow
    Write-Host "    1. Put WLASL_300/ videos in: $(Join-Path $GesturaV2Root 'WLASL_300')" -ForegroundColor White
    Write-Host "    2. Run: $VenvPython $ModelDir\extract_landmarks.py --workers 4" -ForegroundColor White
    Write-Host "    3. Re-run this script" -ForegroundColor White
    Write-Host ""
    Write-Host "  Or specify path explicitly: .\run3_launch.ps1 -LandmarksDir 'C:\path\to\landmarks_300'" -ForegroundColor White
    exit 1
} else {
    Write-Host "  [OK] Landmarks found at: $FoundLandmarks" -ForegroundColor Green
    # Count .npy files
    $npyCount = (Get-ChildItem $FoundLandmarks -Recurse -Filter "*.npy" -ErrorAction SilentlyContinue).Count
    Write-Host "       ($npyCount .npy files)" -ForegroundColor Gray
}

# ── 4. Verify training scripts ─────────────────────────────────────────────
$requiredFiles = @("train.py", "model.py", "dataset.py")
foreach ($f in $requiredFiles) {
    $p = Join-Path $ModelDir $f
    if (-not (Test-Path $p)) {
        Write-Host "  [ERROR] Missing: $p" -ForegroundColor Red
        exit 1
    }
}
Write-Host "  [OK] All training scripts present" -ForegroundColor Green

# ── 5. Set up output dirs ─────────────────────────────────────────────────
$LogDir = Join-Path $ScriptDir "training_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Write-Host "  [OK] Log dir: $LogDir" -ForegroundColor Green

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Run 3 Configuration" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# ── 6. Build training command ──────────────────────────────────────────────
$trainArgs = @(
    "train.py",
    "--landmarks-dir", "`"$FoundLandmarks`"",
    "--epochs", $Epochs,
    "--batch-size", $BatchSize,
    "--num-workers", $NumWorkers,
    "--d-model", "192",
    "--nhead", "6",
    "--num-layers", "3",
    "--dim-ff", "384",
    "--dropout", "0.4",
    "--lr", "5e-4",
    "--min-lr", "1e-6",
    "--weight-decay", "5e-5",
    "--patience", "40",
    "--warmup-epochs", "10",
    "--mixup-alpha", "0.2",
    "--seed", "42"
)

if ($QuickTest) {
    $trainArgs += @("--max-samples", "50", "--epochs", "3")
    Write-Host "  Mode:         QUICK TEST (3 epochs, 50 samples)" -ForegroundColor Magenta
} else {
    Write-Host "  Mode:         FULL TRAINING" -ForegroundColor Green
}

if ($NoMixup) {
    $trainArgs += "--no-mixup"
    Write-Host "  Mixup:        DISABLED" -ForegroundColor Yellow
} else {
    Write-Host "  Mixup:        ON (alpha=0.2)" -ForegroundColor Green
}

if ($AllowCPU -or -not $cudaOK) {
    $trainArgs += "--allow-cpu"
}

if ($Resume -ne "") {
    $trainArgs += @("--resume", "`"$Resume`"")
    Write-Host "  Resume from:  $Resume" -ForegroundColor Yellow
}

Write-Host "  Epochs:       $Epochs"
Write-Host "  Batch size:   $BatchSize"
Write-Host "  Workers:      $NumWorkers"
Write-Host "  Model:        d=192, heads=6, layers=3, ff=384, dropout=0.4 (~1.8M params)"
Write-Host "  LR:           5e-4 → 1e-6 (cosine annealing, 10ep warmup)"
Write-Host "  Label smooth: 0.2 (was 0.05)"
Write-Host "  Early stop:   patience=40 on val_loss (was val_acc)"
Write-Host "  Landmarks:    $FoundLandmarks"
Write-Host "  Checkpoints:  $ModelDir"
Write-Host "  Logs:         $LogDir"
Write-Host ""

# ── 7. Launch training ─────────────────────────────────────────────────────
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Starting Run 3 Training..." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  TIP: If training stops, resume with:" -ForegroundColor Gray
Write-Host "  .\run3_launch.ps1 -Resume '$ModelDir\gesture_model_300.pt'" -ForegroundColor Gray
Write-Host ""

$startTime = Get-Date
Push-Location $ModelDir

try {
    & $VenvPython @trainArgs
    $exitCode = $LASTEXITCODE
} finally {
    Pop-Location
}

$elapsed = (Get-Date) - $startTime
$hours   = [math]::Floor($elapsed.TotalHours)
$mins    = $elapsed.Minutes

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Training finished in ${hours}h ${mins}m (exit code: $exitCode)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "  Next steps:" -ForegroundColor Green
    Write-Host "    1. Analyze results:  $VenvPython $ScriptDir\Results\analyze_results.py" -ForegroundColor White
    Write-Host "    2. Archive results:  $VenvPython $ScriptDir\Results\organize_run.py --run-name Run_3" -ForegroundColor White
    Write-Host ""
}
